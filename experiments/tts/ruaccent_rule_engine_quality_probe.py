#!/usr/bin/env python3
"""Compare RUAccent 1.5.8.3 with its v1.5.7 RuleEngine pipeline restored."""

from __future__ import annotations

import json

from ruaccent import RUAccent
from ruaccent_rule_engine import process_all_internal_with_rule_engine

TEXTS = [
    "У него красивые глаза, суд выписал ордера на обыск, а вы белите потолок каждый год.",
    "Все гости уже пришли.",
    "Всё оборудование уже готово.",
    "У него красивые глаза.",
    "Возле правого глаза появилась царапина.",
    "Голубые озера видны с горы.",
    "На поверхности озера появилась рябь.",
    "Суд выписал ордера на обыск.",
    "Срок действия ордера закончился.",
    "Дом медленно осел после дождя.",
    "Упрямый осел стоял у дороги.",
    "Он сек дрова во дворе.",
    "Подождите пять сек.",
    "Банк выдал новый заем.",
    "Умная колонка включила музыку.",
    "Он берет со стола красный берет.",
    "Лисий помет обнаружили у дерева.",
    "Вы белите потолок каждый год.",
    "Белите потолок аккуратнее!",
    "Вы водите машину очень уверенно.",
    "Водите машину осторожнее!",
    "Вы сушите одежду на балконе.",
    "Сушите одежду на воздухе!",
    "Вы измените настройки после проверки.",
    "Измените настройки прямо сейчас!",
    "Слова матери успокоили ребёнка.",
    "Не материте подчинённых на работе.",
    "На двери висит замок.",
    "Старинный замок стоит на высокой горе.",
    "У дверного замка сломался ключ.",
    "Башни средневекового замка видны издалека.",
    "Я плачу за квартиру каждый месяц.",
    "Я плачу от боли и обиды.",
    "Белки прыгают по веткам.",
    "Белки необходимы организму.",
    "Раздался резкий хлопок.",
    "На поле созрел хлопок.",
    "Кружки стоят на кухонном столе.",
    "Дети записались в спортивные кружки.",
    "Мы купили пшеничную муку.",
    "Он испытывал страшную муку.",
    "Музыкальный орган звучал в соборе.",
    "Этот внутренний орган оказался здоров.",
    "Мы уже приехали.",
    "Этот проход гораздо уже соседнего.",
    "Атлас мира лежит на столе.",
    "Платье сшито из атласа.",
    "Она начала читать книгу.",
    "Все начала координат отмечены на схеме.",
    "Отрасли экономики развиваются неравномерно.",
    "Волосы за лето заметно отрасли.",
]


accent = RUAccent()
accent.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False)

cases = []
for source in TEXTS:
    try:
        baseline = accent.process_all(source)
    except Exception as error:
        baseline = f"{type(error).__name__}: {error}"
    try:
        restored, decisions = process_all_internal_with_rule_engine(accent, source)
    except Exception as error:
        restored = f"{type(error).__name__}: {error}"
        decisions = []
    cases.append(
        {
            "source": source,
            "baseline": baseline,
            "restored": restored,
            "changed": baseline != restored,
            "rule_decisions": decisions,
        }
    )

print(
    json.dumps(
        {
            "cases": len(cases),
            "changed": sum(case["changed"] for case in cases),
            "alignment_errors": sum(
                any(decision.get("alignment_error") for decision in case["rule_decisions"])
                for case in cases
            ),
            "results": cases,
        },
        ensure_ascii=False,
        indent=2,
    )
)
