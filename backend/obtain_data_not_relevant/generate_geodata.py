from Countrydetails import countries, country
import json
from pathlib import Path


def build_geodata():
    world = countries.all_countries()

    countries_set = set(world.countries())
    capitals_set = set(world.capitals().values())

    states_set = set()
    cities_set = set()

    for c in countries_set:
        try:
            details = country.country_details(c)

            states_set.update(details.states() or [])
            cities_set.update(details.cities() or [])

        except Exception:
            # alguns países não têm dados completos
            continue

    return {
        "countries": sorted(countries_set),
        "capitals": sorted(capitals_set),
        "states": sorted(states_set),
        "cities": sorted(cities_set),
    }


def save_python_module(data, output_path: Path):
    """Guarda como módulo Python importável."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated geographic database\n\n")
        for key, values in data.items():
            f.write(f"{key.upper()} = {values}\n\n")


def save_json(data, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

"""
if __name__ == "__main__":
    data = build_geodata()

    save_python_module(data, Path("geodata.py"))
    save_json(data, Path("geodata.json"))

    print("✓ Geodata generated successfully")
"""

from Countrydetails import countries, country

country = countries.all_countries()
# print(country.capitals()) 
# print(country.languages())

from Countrydetails import countries

country = countries.all_countries()
data = country.languages()

# obter todos os códigos de língua
lang_codes = set()

for _, (_, langs) in data.items():
    lang_codes.update(langs)

print(sorted(lang_codes))


import pycountry
from Countrydetails import countries

country = countries.all_countries()
data = country.languages()

lang_codes = set()

for _, (_, langs) in data.items():
    lang_codes.update(langs)

# converter para nomes completos
lang_names = []

for code in lang_codes:
    lang = pycountry.languages.get(alpha_2=code)
    if lang:
        lang_names.append(lang.name)

print(sorted(lang_names))
