#!/usr/bin/env python3
"""Prove whether RUAccent 1.5.8.3 process_all() depends on RuleEngine."""

from __future__ import annotations

import ctypes
import gc
import json
from pathlib import Path

from ruaccent import RUAccent

TEXTS = [
    "Привет, я работаю на Джетсоне.",
    "Я открыл замок старым ключом.",
    "На берегу виднелись старинные замки.",
    "Она начала читать, когда часы пробили полночь.",
    "Начала всех координат лежат на оси.",
    "Косой косой косил косой косой.",
    "Мы купили муку, но я терпеть не могу эту муку.",
    "Орган исполнил сложную партию, а внутренний орган не болел.",
    "Всё уже готово, хотя солнце уже садилось.",
    "Ёжик живёт под ёлкой, а еще любит яблоки.",
    "Поезд отправится двадцать девятого августа в двадцать три часа.",
    "Температура повысилась до тридцати семи целых и двух десятых градуса.",
    "ООО Ромашка заключило договор с банком.",
    "Это новый неизвестный квазитермин для проверки акцентуатора.",
    "Передайте это красивому окну и синему морю.",
]


def rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024, 1)
    return 0.0


def evaluate(accent: RUAccent) -> list[dict[str, str]]:
    results = []
    for text in TEXTS:
        try:
            results.append({"text": text, "status": "ok", "result": accent.process_all(text)})
        except Exception as error:  # The probe must compare identical failures too.
            results.append(
                {
                    "text": text,
                    "status": "error",
                    "result": f"{type(error).__name__}: {error}",
                }
            )
    return results


accent = RUAccent()
accent.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)
loaded_rss = rss_mib()
before = evaluate(accent)

del accent.rule_accent
gc.collect()
ctypes.CDLL("libc.so.6").malloc_trim(0)
without_rule_engine_rss = rss_mib()
after = evaluate(accent)

report = {
    "loaded_rss_mib": loaded_rss,
    "without_rule_engine_rss_mib": without_rule_engine_rss,
    "released_mib": round(loaded_rss - without_rule_engine_rss, 1),
    "exact_match": before == after,
    "successful_cases": sum(item["status"] == "ok" for item in after),
    "failed_cases": sum(item["status"] == "error" for item in after),
    "results": after,
}
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["exact_match"] else 1)
