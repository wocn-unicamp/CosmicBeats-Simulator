import collections
import os
import time
import requests

from src.schedulers.llm_scheduler import (MAX_ATTEMPTS, TRANSIENT_HTTP, backoff_s,
                                         build_prompt, parse_answer, response_meta)
import urllib3
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY        = os.environ.get("GEMINI_API_KEY")
SLM_MODEL      = os.environ.get("SLM_MODEL", "gemma-4-26b-a4b-it")
NPU_LATENCY_MS = float(os.environ.get("SLM_NPU_LATENCY_MS", "50.0"))
RPM_LIMIT      = int(os.environ.get("SLM_RPM_LIMIT", "14"))  # conservative under the 15 RPM cap

# Protocolo v3 (27/09/2026, docs Parte X). O SLM recebe o MESMO prompt do LLM e
# responde no mesmo formato (satellite_id), lido pela mesma funcao. Unicas
# diferencas, ambas por necessidade do modelo e nao por escolha de desenho:
#  - a Gemma raciocina antes de responder, entao so as partes finais (sem
#    "thought") sao lidas; o raciocinio nunca e usado como decisao;
#  - o orcamento de saida precisa cobrir o raciocinio: 1024 tokens cortavam a
#    resposta em 13 de 13 casos testados, e 4096 ainda cortou 2 (IX.9.3).
MAX_OUTPUT_TOKENS = 8192
REQUEST_TIMEOUT_S = 120   # raciocinio longo; o LLM, sem raciocinio, usa 60 s


class SLMScheduler:
    def __init__(self):
        if not API_KEY:
            print(">>> [SLM] WARNING: GEMINI_API_KEY not found. API calls will return None (drop).")
        else:
            print(f">>> [ENGINE] SLM Scheduler Initialized ({SLM_MODEL} via Gemini API | NPU latency={NPU_LATENCY_MS}ms)")
        self.npu_busy_until = 0.0
        self.pending_tasks = []
        self._call_timestamps: collections.deque = collections.deque()

    # ------------------------------------------------------------------ #
    # Async state machine                                                  #
    # ------------------------------------------------------------------ #

    def receive_task(self, task_dict, current_time, fleet):
        ready_time = max(current_time, self.npu_busy_until) + (NPU_LATENCY_MS / 1000.0)
        self.npu_busy_until = ready_time
        task_dict["slm_ready_time"] = ready_time
        task_dict["slm_fleet_snapshot"] = list(fleet)
        self.pending_tasks.append(task_dict)
        return ready_time

    def check_completed_inferences(self, current_time):
        completed, remaining = [], []
        for task in self.pending_tasks:
            if current_time >= task.get("slm_ready_time", 0):
                completed.append((task, self._decide_route(task)))
            else:
                remaining.append(task)
        self.pending_tasks = remaining
        return completed

    # ------------------------------------------------------------------ #
    # Decision logic                                                       #
    # ------------------------------------------------------------------ #

    def _decide_route(self, task_dict):
        fleet = task_dict.get("slm_fleet_snapshot", [])
        t0 = time.perf_counter()
        parsed = self._call_gemini(task_dict, fleet)
        # Falha de API ou de parsing NAO e decisao do modelo: registrada a parte.
        task_dict["decision_source"] = ("model" if parsed is not None
                                        else getattr(self, "_last_failure", "api_failure"))
        task_dict["slm_wall_latency_ms"] = (time.perf_counter() - t0) * 1000
        # Como no LLM: o satellite_id e devolvido tal como o modelo o deu.
        return parsed.get("satellite_id") if parsed is not None else None

    def _rate_limit_wait(self):
        """Block until making another API call stays within RPM_LIMIT calls/minute."""
        now = time.monotonic()
        while self._call_timestamps and now - self._call_timestamps[0] > 60.0:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= RPM_LIMIT:
            wait = 61.0 - (now - self._call_timestamps[0])
            if wait > 0:
                print(f"   [SLM Rate Limiter] {len(self._call_timestamps)} calls/min — waiting {wait:.1f}s")
                time.sleep(wait)
            now = time.monotonic()
            while self._call_timestamps and now - self._call_timestamps[0] > 60.0:
                self._call_timestamps.popleft()
        self._call_timestamps.append(time.monotonic())

    def _call_gemini(self, task_dict, fleet):
        """Devolve a resposta decodificada, ou None (com _last_failure indicando a causa)."""
        self._last_failure = "api_failure"
        if not API_KEY:
            return None
        self._rate_limit_wait()
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{SLM_MODEL}:generateContent?key={API_KEY}")
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": build_prompt(task_dict, fleet)}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        }
        for attempt in range(MAX_ATTEMPTS):
            try:
                t0 = time.perf_counter()
                r = requests.post(url, headers=headers, json=data, verify=False,
                                  timeout=REQUEST_TIMEOUT_S)
                latencia_ms = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    body = r.json()
                    task_dict.update(response_meta(body))
                    parts = (body.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
                    # So a resposta final. Se o modelo nao chegou a responder (por
                    # exemplo, raciocinio cortado pelo orcamento), e falha de parse:
                    # o texto do raciocinio nunca vira decisao.
                    raw = "".join(p.get("text", "") for p in parts if not p.get("thought", False))
                    parsed = parse_answer(raw) if raw.strip() else None
                    if parsed is None:
                        print(f"   [SLM Parse Error] finish={task_dict.get('finish_reason')}"
                              f" — sem JSON valido na resposta final: {raw[:80]!r}")
                        self._last_failure = "parse_failure"
                        return None
                    print(f"   [SLM {SLM_MODEL} {latencia_ms:.0f}ms] satellite_id="
                          f"{parsed.get('satellite_id')} reason={parsed.get('reason', '')}")
                    return parsed
                elif r.status_code in TRANSIENT_HTTP:
                    wait = backoff_s(attempt, r.status_code)
                    print(f"   [SLM HTTP {r.status_code}] Tentativa {attempt+1}/{MAX_ATTEMPTS}"
                          f" — aguardando {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    print(f"   [SLM Error] HTTP {r.status_code} — {r.text[:120]}")
                    break
            except Exception as e:
                print(f"   [SLM Connection Error] {e}")
                time.sleep(backoff_s(attempt))
                continue
        return None
