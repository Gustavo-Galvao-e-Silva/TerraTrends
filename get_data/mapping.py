# qcew_industry_map.py

TARGET_INDUSTRIES = {
    "Accommodation and food services": ["72"],
    "Administrative and support and waste management and remediation services": ["56"],
    "Agriculture, forestry, fishing and hunting": ["11"],
    "Arts, entertainment, and recreation": ["71"],
    "Construction": ["23"],
    "Durable goods manufacturing": [  # refine with specific NAICS 3-digit manufacturing codes
        # e.g. "321", "327", "331", ...
    ],
    "Educational services": ["61"],
    "Finance and insurance": ["52"],
    "Government and government enterprises": [
        # QCEW typically uses ownership + industry code:
        # you'll filter with ownership codes later (e.g. 1,2,3 for private/state/local/federal),
        # and industry "92" for public admin; leave list empty and handle via special logic if needed
    ],
    "Health care and social assistance": ["62"],
    "Information": ["51"],
    "Natural resources and mining": ["11", "21"],
    "Nondurable goods manufacturing": [
        # complement set of durable manufacturing NAICS in 31-33
    ],
    "Other services (except government and government enterprises)": ["81"],
    "Professional and business services": ["54", "55", "56"],
    "Real estate and rental and leasing": ["53"],
    "Retail trade": ["44", "45"],
    "Transportation and warehousing": ["48", "49"],
    "Utilities": ["22"],
    "Wholesale trade": ["42"],
}
