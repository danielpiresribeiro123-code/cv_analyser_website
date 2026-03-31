import re
from gliner import GLiNER
import pandas as pd
from typing import Union, List, Dict, Optional, Any, cast, Tuple, Set, Callable
from pathlib import Path
from datetime import datetime

CURRENT_DATE = datetime.now()
from services.paths_models_and_data import get_gliner_model, SKILL_LABELS, COUNTRIES_AND_CAPITALS, COUNTRIES_NAMES, CAPITALS_NAMES, LANGUAGES, OUTPUTS_DIR

# ----------------------------------------------------------
# Performance / fallback flags
# ----------------------------------------------------------
USE_REGEX_FALLBACKS = True

USE_JOB_TITLE_REGEX_FALLBACK = True
USE_SKILL_REGEX_FALLBACK = True

USE_SOFTWARE_REGEX_FALLBACK = True
USE_PROGRAMMING_REGEX_FALLBACK = True
USE_GENERAL_SKILLS_REGEX_FALLBACK = False   # este costuma ser o mais caro e mais ruidoso

USE_REGEX_FALLBACKS_ON_CHUNKS = False       # importante para performance


# ----------------------------------------------------------
# Gliner labels, regex, etc
# ----------------------------------------------------------

SIMPLE_JOB_TITLES = {
    # EN
    "teacher", "professor", "manager", "analyst", "engineer", "developer",
    "consultant", "coordinator", "supervisor", "assistant", "technician",
    "driver", "lifeguard", "operator", "pilot", "scientist", "private security",
    "security officer", "security guard", "deliveryman", "forester","biochemist",
    "food scientist", "geneticist","farmer", "biologist", "horticulturalist",
    "inspector","beekeeper","ecologist","conservationist", "veterinarian",
    "zoologist", "groomer", "animal control officer", "attendant", "pet walker",
    "animal sitter", "breeder", "pathologist","ophthalmologist", "reporter",
    "actuary","clerk", "artist", "therapist", "barber", "beautician", "designer",
    "stylist","cashier", "coordinator", "advisor", "writer", "editor", "illustrator",
    "actor","singer","producer","web developer","architect","animator", "painter",
    "tutor", "headteacher", "counsellor", "nurse", "doctor", "personal trainer",
    "pharmacist","dentist", "orthodontist", "dietitian", "housekeeper", "porter",
    "tour guide", "supervisor", "director", "makeup artist","hairstylist","lecturer",
    "professional staff","engineer","intern","developer","assistant","captain",
    "analyzer","wardrobe stylist","chairman","accountant","fire fighter","sailor",
    "co-ordinator","coordinator","orthopaedics","plumber","account assistant",
    "quality control & testing department","quality control & testing engineer",
    "quantity survey assistant","computer operator & store department","assistant Engineer",
    "head of the department","house captain","orthopaedic surgeon","surgeon","technicians",
    "technician","secretary","site consultants","super admin","super-admin","admin",
    "referees","Manicurist","assistant manager", "hunter","Hairstylist & Makeup Artist",

    # First 500
    'Assistant General Manager', 'Business Development Executive', 'Cashier/Accountant', 
    'Civil Supervisor', 'Document Controller', 'Draughtsman', 'Duct Charge Hand', 'Duct Man', 
    'Electrical Supervisor', 'Founder', 'Graduate Engineer Trainee', 'HR Assistant', 'HR Executive', 
    'HSE Officer', 'Instructor', 'Junior Engineer', 'Junior Executive', 'Junior Specialist', 
    'Logistics/Store Executive', 'Locum Specialist', 'Maintenance Engineer', 'Manager MEP Services', 
    'Marketing Executive', 'Medical Doctor', 'Medical Officer', 'Medical Representative', 'MEP Head', 
    'MIS Executive', 'Office Executive', 'Road Surveyor Assistant', 'Site In Charge', 'Store Keeper', 
    'Technical Coordinator', 'Technical Manager', 'Volunteer Assistant Accountant', 'Worker',

    # 500 - 1000
    'AC Technician', 'Account & Finance Executive', 'Account Manager', 'Account Officer', 
    'Accounts Assistant', 'Admin Officer', 'Administration Officer', 'Advocate', 'Aluminium Fabricator', 
    'Assistant Production Manager', 'Credit Analyst', 'Electrical Design Engineer', 'Field Engineer', 
    'Field Survey Officer', 'HVAC Technician', 'IT Support Engineer', 'Maintenance Engineer', 'Manager', 
    'Production Engineer', 'Production Manager', 'Sales Executive', 'Salesman', 'Senior Executive', 
    'Shift Engineer', 'Site Manager', 'Specialist Intern', 'Store Executive', 'Teaching Assistant', 
    'Trainee Engineer', 'Transmission Engineer', 'Vice President', 'Web Designer', 'Web Designer & Developer',

    # 1000 - 1500
    'Account Coordinator', 'Accounts Assistant', 'Administration Assistant', 'Area Sales Manager', 
    'Article Clerk', 'Assistant Commandant', 'Assistant Lecturer', 'Assistant Manager', 'Assistant Site Engineer', 
    'Assistant Storekeeper', 'Audit Assistant', 'Auditor', 'Back Office Executive', 'Banking Training Manager', 
    'BI Consultant', 'Branch Engineer', 'Business Development Executive', 'CAD Draughtsman', 'Civil Engineer', 
    'Commercial Officer', 'Deputy Manager', 'Duty Manager', 'Electrical Engineer', 'Engineer', 'Field Assistant', 
    'Financial Manager', 'HSE Officer', 'HVAC Engineer', 'IT Engineer', 'Lecturer', 'Logistics Coordinator', 
    'Maintenance Engineer', 'Mechanical Engineer', 'Office Assistant', 'Portfolio Manager', 'Production Manager', 
    'Quality Engineer', 'RF Engineer', 'Sales Assistant', 'Scaffolding Coordinator', 'Scaffolding Specialist', 
    'Security Officer', 'Shift Engineer', 'Site Engineer', 'Station Manager', 'Store Manager', 'Technical Assistant', 
    'Technician', 'Telecommunication Engineer', 'Test Manager', 'Testing Engineer', 'Trainee Engineer', 'Web Designer',

    # 1500 - 2000
    'Accountant', 'Assistant Manager', 'Assistant Site Engineer', 'Cashier', 'Celebrity Makeup Artist', 'Chairman', 
    'Chief Accountant', 'Chief Engineer', 'Civil Draughtsman', 'Civil Engineer', 'Civil Foreman', 'Civil Quantity Surveyor', 
    'Civil Site Supervisor', 'Clerk', 'Client Executive', 'Commercial Executive', 'Commissioning Engineer', 'Company Secretary',  
    'Consultant Engineer', 'Consultant Orthopaedic Surgeon', 'Coordinator', 'Cost Accountant', 'County Administrator', 
    'Engineer', 'Manager', 'Production Manager', 'Sales Manager', 'Security Officer', 'Site Supervisor', 'Store Manager', 'Technician',

    # 2000 - 2500
    'Accounts Executive', 'Assistant Instruments Technician', 'Customer Relationship Executive', 'Dean of Students', 'Deputy Finance Manager', 
    'Deputy Manager', 'Deputy Team Leader', 'Design Consultant', 'Design Engineer', 'Desk Executive', 'Development Manager', 'Diesel Mechanic', 
    'Diploma Engineer', 'Diploma Engineer Trainee', 'Director', 'Distinguished Professor', 'Document Controller', 'Draftsman', 'Economist', 
    'Editor in Chief', 'Editorial Board Member', 'Electrical Engineer', 'Electrical Instrument Officer', 'Electrical Supervisor', 'Electrician', 
    'Emergency Medical Officer', 'Engineer', 'Estimation Engineer', 'Executive', 'Finance Manager', 'Instruments Technician', 'Manager', 
    'Orthopedic Surgeon', 'Piping Stress Engineer', 'Production Manager', 'Professor', 'Relation Ship Manager', 'Secretary', 'Service Engineer', 
    'Shift In-Charge', 'Site Engineer', 'Supervisor', 'Teaching Assistant', 'Technician', 'Team Leader',

    # 2500 - 3000
    'Entrepreneur', 'Enumerator', 'Engineer Trainee', 'Engineering Consultant', 'Executive Secretary', 'Excise Clerk', 'Exercise Grader', 
    'Facility Executive', 'Factory Manager', 'Faculty Member', 'Finance Executive', 'Fire and Security Officer', 'Fitter', 'Foreman', 
    'Foundation Doctor', 'Fund Administrator', 'General Accountant', 'General Manager', 'General Practitioner', 
    'Global Lead Knowledge Manager', 'Glims Administrator',

    # 3000 - 3500
    'Government Apprentice Trainee', 'Graduate Engineer', 'Guest Relation Officer', 'HSE Officer', 'Hair Stylist', 'Hand Surgeon', 
    'Head of Department', 'Head Lecturer', 'Helpdesk Consultant', 'Helper Electrical', 'Housekeeping Team Leader', 'HR Manager', 'HR Officer', 
    'HR Recruiter', 'HR Trainee', 'HVAC Consultant', 'HVAC Professional', 'Incident Controller', 'Industrial Trainee', 'Installation Technician', 
    'Instrument Technician', 'Instructor', 'Intern Trainee', 'Junior Electrical Engineer', 'Logistics Coordinator', 'Manager', 'Medical Practitioner', 
    'Orthopedic Surgeon', 'Receptionist', 'Security Officer', 'Site Engineer', 'Supervisor', 'Technician', 'Team Leader',

    # 3500 - 4000
    'IT Consultant', 'IT Executive', 'IT Secretary', 'Junior Accountant', 'Junior Web Developer', 'Junior Maintenance Engineer', 'Junior Production Executive', 
    'Junior Consultant', 'Lab Instructor', 'Lathe Machine Operator', 'Legal Intern', 'Logistics Coordinator', 'Logistics Assistant', 'Maintenance Engineer', 
    'Maintenance Manager', 'Maintenance Technician', 'Manager', 'Marketing Manager', 'NDT Technician', 'Officer', 'Orthopaedic Surgeon', 'Registrar', 
    'Service Engineer', 'Site Engineer', 'Specialist Registrar', 'Technician',

    # 4000 - 4500
    'Major Trauma Consultant', 'Course Coordinator', 'Make Up Artist', 'Makeup Artist & Hairstylist', 'Makeup Intern', 'Management Trainee', 'Manager', 
    'Safety Officer', 'Marketing Manager', 'Mechanical Foreman', 'Mechanical Technician', 'Media Group Leader', 'Medical Officer', 'Mens Groomer', 
    'MEP Designer Engineer', 'Meter Reader', 'Middle Level Executive', 'Mobile Sales Manager', 'Module Leader', 'Lecturer', 'Motivational Speaker', 
    'Trainee Engineer', 'Piping Stress Engineer', 'Junior Engineer', 'Maintenance Engineer', 'Environment Engineer', 'Warehouse Assistant', 'NDT Technician', 
    'QC Technician', 'Neurosurgeon',

    # 4500 - 5000
    'Office Assistant', 'Office Coordinator', 'Safety Officer', 'IT Consultant', 'IT Executive', 'IT Secretary', 'Onsite/Offshore Coordinator', 
    'Orthopaedic Surgeon', 'Orthopedic Consultant', 'Orthopedic Surgery Registrar', 'Personal Assistant', 'Personal Accountant', 'Piping Design Engineer', 
    'Piping Designer', 'Planning Engineer', 'Plant Manager', 'Plastic Surgeon', 'Restaurant Manager', 'Sales Retail Executive', 'Site Engineer', 'Store Officer', 
    'Supervisor', 'Technician', 'Testing Engineer', 'Medical Officer', 'Executive Engineer', 'Cashier', 'Consultant',

    # 5000 - 5500
    'Presales Engineer', 'President', 'Production Engineer', 'Production Supervisor', 'Procurement Executive', 'Process Developer', 'Programmer', 'Project Engineer', 
    'QA/QC Engineer', 'QC Inspector', 'Quality Engineer', 'Quality Executive', 'Quality Head', 'Quantity Surveyor', 'Radio Officer', 'Regional Manager', 'Registrar', 
    'Relationship Manager', 'Repair Technician', 'Report Writer', 'Resident Engineer', 'Resident Registrar', 'Residential Medical Officer', 'Restaurant Cashier', 
    'Retail Banking Officer', 'Reviewer',

    # 5500 - 6000
    'Rig Mechanic', 'Safety Engineer', 'Sales Consultant', 'Sales Representative', 'Scaffolder', 'Scaffolding Foreman', 'Scaffolding Inspector', 'Security Executive', 
    'Security Officer', 'Senior Consultant', 'Senior Design Engineer', 'Senior Medical Officer', 'Senior Piping Designer', 'Senior Safety Officer', 
    'Senior Scaffolding Supervisor', 'Senior Site Engineer', 'Service Engineer', 'Shift In-Charge', 'Site Electrical Engineer', 'Site Engineer', 'Site Inspector', 
    'Site Supervisor', 'Site Technician', 'Software Developer', 'Software Tester', 'Solar Energy Installer', 'Solar Thermal Energy Engineer', 'Specialist Registrar', 
    'Specialty Doctor', 'Spine Surgeon', 'Store Keeper', 'Supervisor', 'Technician',

    # 6000 - 6500
    'Sr. General Manager', 'Sr. HR Executive', 'Sr. Production Manager', 'Sr. Telecom Technician', 'Staff Neurosurgeon', 'Store Assistant', 'Store Incharge', 
    'Store Manager', 'Store Officer', 'Store Supervisor', 'Structure Consultant', 'Sub Investigator', 'Summer Trainee', 'Supervisor', 'Support Engineer', 
    'Support Executive', 'System Integrator', 'Tax Department Intern', 'Table Instructor', 'Tactical Security Officer', 'TBM Manager', 'Team Head', 'Team Lead', 
    'Technical Assistant', 'Technical Support Coordinator', 'Technical Voice Support', 'Technical Writer', 'Technician', 'Telecaller', 'Telecom Technician', 
    'Test Engineer', 'Testing & Commissioning Engineer', 'Testing Engineer', 'QA/QC Manager', 'Quality Control Specialist', 'Commercial Officer', 'Construction Manager', 
    'HSE Manager', 'General Manager', 'Plant Engineer', 'Lab Technician', 'Site Manager',

    # 6500 - 7000
    'Tower Officer', 'Tower Officer Supervisor', 'Trainee Engineer', 'Trainee Sonographer', 'Training Coordinator', 'Training Manager', 'Service Engineer', 
    'Estimation Engineer', 'Accounts Assistant', 'Teaching Assistant', 'VA Developer', 'Vice-President', 'Visiting Consultant', 'Visiting Faculty', 
    'Voluntary Medical Officer', 'Warehouse Assistant Supervisor', 'Welder', 'Administrative Assistant', 'Audit Assistant', 'Civil Engineer', 'Engineer', 
    'HVAC Engineer', 'Junior Engineer', 'Locum Specialist', 'Medical Officer', 'Office Assistant', 'Quality Engineer', 'Relationship Manager', 'Shift Engineer', 
    'Site Manager', 'Branch Manager', 'Mechanical Engineer', 'MEP Engineer', 'Room Technician', 'Sales Manager', 'Document Controller', 'Sales Executive', 
    'Workshop In Charge', 'Yard Manager', 'Manager',
    
    # PT
    "professor", "professora", "gestor", "gerente", "analista", "engenheiro",
    "engenheira", "desenvolvedor", "desenvolvedora", "consultor", "consultora",
    "coordenador", "coordenadora", "supervisor", "supervisora", "assistente",
    "técnico", "técnica", "motorista", "condutor", "nadador-salvador",
    "operador", "operadora", "piloto", "segurança",

    # ES
    "profesor", "profesora", "gerente", "analista", "ingeniero", "ingeniera",
    "desarrollador", "desarrolladora", "consultor", "consultora", "coordinador",
    "coordinadora", "supervisor", "supervisora", "asistente", "técnico",
    "técnica", "conductor", "socorrista", "operador", "operadora", "piloto",
    "seguridad",
}

SIMPLE_JOB_TITLE_REGEX = re.compile(
    rf"\b(?:{'|'.join(sorted(map(re.escape, SIMPLE_JOB_TITLES), key=len, reverse=True))})\b",
    re.IGNORECASE
)

COMPOUND_JOB_TITLE_PATTERNS = [
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}manager\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}coordinator\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}assistant\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}driver\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}technician\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}officer\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}guard\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}specialist\b",
    r"\b(?:[a-zA-ZÀ-ÖØ-öø-ÿ]+\s+){1,3}engineer\b",
    r"\bprivate security\b",
    r"\boperations coordinator\b",
    r"\bsite construction preparation manager\b",
    r"\bsite construction preparation assistant\b",
]

COMPOUND_JOB_TITLE_REGEXES = [
    re.compile(pattern, re.IGNORECASE) for pattern in COMPOUND_JOB_TITLE_PATTERNS
]

# Define label sets globally for reuse
DEGREE_LEVELS = [
    "high school",
    "associate",
    "bachelor",
    "master",
    "doctor",
    "doctorate",
    "phd",
]

DEGREE_NORMALIZATION_MAP = {
    # --------------------
    # Secondary / High School
    # --------------------
    "highschool": "High School Diploma",
    "high school": "High School Diploma",
    "secondary school": "High School Diploma",
    "hs diploma": "High School Diploma",
    "diploma do ensino secundário": "High School Diploma",
    "ensino secundário": "High School Diploma",
    "12º year": "High School Diploma",
    "12º ano": "High School Diploma",
    "12o ano": "High School Diploma",

    # --------------------
    # Associate / Short-cycle
    # --------------------
    "associate": "Associate Degree",
    "a.a": "Associate of Arts",
    "aa": "Associate of Arts",
    "a.s": "Associate of Science",
    "as": "Associate of Science",
    "cets": "Short-cycle Higher Education (CTeSP)",
    "ctesp": "Short-cycle Higher Education (CTeSP)",
    "curso técnico superior profissional": "Short-cycle Higher Education (CTeSP)",

    # --------------------
    # Bachelor
    # --------------------
    "bachelor": "Bachelor’s Degree",
    "bachelors": "Bachelor’s Degree",
    "undergraduate": "Bachelor’s Degree",
    "graduação": "Bachelor’s Degree",
    "licenciatura": "Bachelor’s Degree",
    "licenciado": "Bachelor’s Degree",

    # Bachelor abbreviations
    "b.sc": "Bachelor of Science",
    "bsc": "Bachelor of Science",
    "b.s": "Bachelor of Science",
    "bs": "Bachelor of Science",
    "b.eng": "Bachelor of Engineering",
    "beng": "Bachelor of Engineering",
    "b.e": "Bachelor of Engineering",
    "be": "Bachelor of Engineering",
    "b.tech": "Bachelor of Technology",
    "btech": "Bachelor of Technology",
    "b.a": "Bachelor of Arts",
    "ba": "Bachelor of Arts",
    "bba": "Bachelor of Business Administration",
    "llb": "Bachelor of Laws",

    # --------------------
    # Master
    # --------------------
    "master": "Master’s Degree",
    "masters": "Master’s Degree",
    "postgraduate": "Master’s Degree",
    "pós-graduação": "Postgraduate Qualification",
    "pos-graduação": "Postgraduate Qualification",
    "pos graduacao": "Postgraduate Qualification",
    "pós graduação": "Postgraduate Qualification",
    "mestrado": "Master’s Degree",
    "mestre": "Master’s Degree",

    # Master abbreviations
    "m.sc": "Master of Science",
    "msc": "Master of Science",
    "m.s": "Master of Science",
    "ms": "Master of Science",
    "m.eng": "Master of Engineering",
    "meng": "Master of Engineering",
    "m.e": "Master of Engineering",
    "me": "Master of Engineering",
    "m.tech": "Master of Technology",
    "mtech": "Master of Technology",
    "m.a": "Master of Arts",
    "ma": "Master of Arts",
    "mba": "Master of Business Administration",
    "llm": "Master of Laws",

    # Integrated Master (PT)
    "mestrado integrado": "Integrated Master’s Degree",

    # --------------------
    # Doctorate
    # --------------------
    "doctorate": "Doctorate",
    "doctoral": "Doctorate",
    "doutoramento": "Doctorate",
    "doutorado": "Doctorate",

    # Doctorate abbreviations
    "ph.d": "Doctor of Philosophy",
    "phd": "Doctor of Philosophy",
    "d.phil": "Doctor of Philosophy",
    "dphil": "Doctor of Philosophy",
    "md": "Doctor of Medicine",
    "m.d": "Doctor of Medicine",
    "jd": "Juris Doctor",
    "j.d": "Juris Doctor",
}

SOFTWARE_AS_SKILL = {
    # Productivity / Office
    "microsoft office", "office", "microsoft", "ms office", 
    "excel", "word", "powerpoint", "outlook", "onenote",
    "google docs", "google sheets", "google slides", "google drive",
    "libreoffice", "wps office", "ms-paint","ms paint",
    "ms excel", "ms word", "ms powerpoint", "ms outlook", "ms onenote",

    # Operating systems / basic tools
    "windows", "window", "win7", "winxp", "win10", "win11",
    "linux", "ubuntu", "debian", "fedora", "centos", "red hat",
    "macos","windows 8","windows 9","windows 10","windows 11","windows 7","windows XP",
    "windows8","windows9","windows10","windows11","windows7","windowsXP",
    "window8","window9","window10","window11","window7","windowXP",
    "window 8","window 9","window 10","window 11","window 7","window XP",
    "winrar", "win-rar", "7zip", "7-zip",

    # Design / CAD / 3D
    "autocad", "auto cad","solidworks","solid works","catia", "revit", "sketchup", "fusion 360",
    "blender", "3ds max", "maya", "cinema 4d", "rhino", "rhinoceros",
    "inventor", "archicad", "microstation", "nozzle pro", "nozzlepro", 

    # Graphic design / UI / video
    "canva", "figma", "adobe xd",
    "photoshop", "adobe photoshop",
    "illustrator", "adobe illustrator",
    "indesign", "adobe indesign", "photo shop",
    "premiere pro", "adobe premiere pro",
    "after effects", "adobe after effects",
    "lightroom", "adobe lightroom",
    "davinci resolve", "final cut pro", "capcut",

    # Development environments / editors
    "vscode", "visual studio code", "visual studio",
    "pycharm", "intellij", "intellij idea",
    "eclipse", "netbeans", "codeblocks", "code::blocks",
    "android studio", "xcode", "spyder", "jupyter", "jupyter notebook",
    "notepad++", "sublime text",

    # Version control / Dev tools
    "git", "github", "gitlab", "bitbucket",
    "docker", "kubernetes", "jenkins", "ansible", "terraform",
    "postman", "swagger",

    # Databases / data tools
    "mysql", "postgresql", "postgres", "sql server", "mssql",
    "oracle", "sqlite", "mongodb", "access", "microsoft access",
    "power bi", "tableau", "qlik", "qlikview", "looker",
    "sas", "spss", "stata", "minitab", "matlab",

    # Accounting / ERP / business software
    "sap erp", "sap hana","sap",
    "oracle erp", "odoo", "primavera", "sage", "tally",
    "crm", "erp", "salesforce", "hubspot",

    # Engineering / scientific
    "ansys", "comsol", "labview", "simulink", "originlab",
    "etabs", "robot structural analysis",

    # Project / collaboration tools
    "jira", "confluence", "trello", "asana", "notion",
    "slack", "microsoft teams", "teams", "zoom",

    # Web / CMS / marketing
    "wordpress", "shopify", "woocommerce",
    "google analytics", "meta business suite",

    # GIS / mapping
    "arcgis", "qgis",

    # Cybersecurity / networking
    "wireshark", "cisco packet tracer", "packet tracer", "nmap", "xampp",
    
    
    # extra software / technical tools
    "arduino",
    "navis works",
    "navisworks",
    "psim",
    "pv-elite",
    "pv elite",
    "caesar ii",
    "caesar-ii",
    "caesar ii piping stress analysis",
    "tally 9.0",
    "tally erp 9",
    "ms-office",
    "ms office",}

PROGRAMMING_LIBRARIES = {
    # ----------------------
    # Programming Languages
    # ----------------------
    # Core / general purpose
    "java", "python", "c", "c#", "c++", "rust", "assembly",

    # Web
    "javascript", "typescript", "html", "css", "sass", "scss",

    # Backend / enterprise
    "golang", "kotlin", "scala", "groovy",

    # .NET
    "vb.net", "f#", ".net",

    # Scripting
    "bash", "shell", "powershell", "perl",

    # Data / scientific
    "matlab", "octave", "julia",

    # Mobile
    "dart", "objective-c", "swift",

    # Functional
    "haskell", "ocaml", "erlang", "elixir", "clojure",

    # Legacy / others
    "fortran", "cobol", "pascal", "ada",

    # Game dev / niche
    "gdscript", "haxe",

    # Hardware / low-level
    "verilog", "vhdl",

    # Query / data languages
    "sql", "pl/sql", "t-sql","sql ",

    # NoSQL / query-like
    "cypher", "graphql",

    # Common misspellings / variants
    "pyhton", "javscript", "c plus plus", "c sharp",
    # ---------------------
    # Python - Data Science / ML
    # ---------------------
    "numpy", "pandas", "scipy", "matplotlib", "seaborn",
    "scikit-learn", "sklearn",
    "statsmodels",
    "xgboost", "lightgbm", "catboost",
    "tensorflow", "keras", "pytorch", "torch",
    "fastai","plotly", "bokeh","streamlit",

    # ---------------------
    # Python - Backend / APIs
    # ---------------------
    "django", "flask", "fastapi",
    "sqlalchemy", "alembic",
    "pydantic",

    # ---------------------
    # Python - Utils
    # ---------------------
    "beautifulsoup", "beautifulsoup4", "bs4",
    "pytest","celery", "beautiful soup",

    # ---------------------
    # JavaScript / Frontend
    # ---------------------
    "react.js", "reactjs",
    "vue", "vue.js", "vuejs",
    "angular", "angularjs",
    "svelte",
    "next.js", "nextjs",
    "nuxt.js", "nuxtjs",

    # UI frameworks
    "bootstrap", "tailwind", "tailwindcss",
    "material-ui",
    "chakra-ui",

    # ---------------------
    # JavaScript / Backend
    # ---------------------
    "node.js", "nodejs",
    "express.js",
    "nestjs",
    "socket.io",

    # ---------------------
    # Java / JVM
    # ---------------------
    "spring boot", "springboot",
    "hibernate",

    # ---------------------
    # .NET
    # ---------------------
    ".net", "asp.net", "aspnet",
    "entity framework", "ef core",

    # ---------------------
    # PHP
    # ---------------------
    "laravel", "symfony", "codeigniter",

    # ---------------------
    # Ruby
    # ---------------------
    "ruby on rails", "rails",

    # ---------------------
    # Databases / ORMs / connectors
    # ---------------------
    "mysql connector", "mysql-connector", "mysqlextra",
    "psycopg2",
    "sqlite3",
    "mongodb driver", "mongoose",
    "redis",

    # ---------------------
    # Data engineering
    # ---------------------
    "pyspark", 
    "airflow",

    # ---------------------
    # Testing
    # ---------------------
    "jest", "mocha", "chai",
    "junit",
    "selenium",

    # ---------------------
    # Misc / common libs
    # ---------------------
    "lodash",
    "axios",
    "moment.js", "moment",
    "three.js", "d3.js", "chart.js",
    
    
    # engineering / technical concepts
    "deep learning", "CNN", "NLP", "deeplearning", "machinelearning",
    "machine learning","computer science",
    "fault tree analysis",
    "harmonic analysis",
    }

GENERAL_SKILLS ={
    # engineering / technical concepts
    "short circuit","voltage drop calculation","microprocessor based relay","numerical relay","circuit breaker","circuit breaker panel board","welding","dc welding","brazing",
    "fabricator","method statement","preliminary stress analysis","soil testing","initial final testing","concrete consistence test","road levelling",
    "pipe line","pipeline","reinforcement","foundation","shuttering","floor finishing","finishing work","plate rolling","marketing","finance","networking","driving license",
    "flexible", "Fixing", "excellent communication","organize", "supervising","games","sciences","reliability",
    
    # specific useful skills / methodologies
    "8d approach","6sigma","5s activities",
    "cpk","poka yoke technique",

    # usual adjectifs
    "poka yoke","poka-yoke","poka -yoke", "poka-yoke", "creative",
    "confident","positive thinking","responsible","sincere","good listener","hard worker","risk assessment","site execution","team worker",
    "disciplined","net surfing","ambitious","fire fighting","firefighting","co-ordination","conduiting","confident and determined",
    "coordination","confident", "determined","IT proficiency","team member","leader","good leader","team leader",
    "internet browsing","arthroplasty","loop checking","suggestive selling","plastering","maintenance team","active listener","lathe",
    "leadership quality","handling quick books", "making friends", "self-confident","multi-tasking","machine maintenance","manage",
    "keen observer","server side technologies", "IT forte","duct sizing", "thermal testing", "Electrical internal & External Electrification",
    "TRT testing","driving","good communication","honest","artistic","banking","brand maintenance","teaching","housing","respect","availability",
    "team leader","coordinate","smart","drive","friendly","carrying","tenure","filling","dedicated","positive attitude","power distribution",
    "efficient","motivated","learner","trading","positive attitude","preventive","preventive & maintenance","critical thinker","cross-team","department head",
    "civil helper","civil maintenance","breakdown maintenance","supervise","analyze",

    # First 500
    'account reconciliation', 'accounts receivable', 'AutoCAD', 'AVR Studio', 'BMS interface', 'bookkeeping', 'C', 'C++', 'cash book', 'close loop checking', 
    'CMD software', 'CNC', 'confined space entry', 'CoralDRAW', 'Coral Video Studio', 'design & layout', 'detailed estimation', 'DGPS', 'drafting software', 
    'earth resistance testing', 'electrical panel maintenance', 'erection & commissioning', 'fire prevention & fire fighting', 'First Aid', 'fixed assets', 
    'front end', 'function test', 'GD&T', 'H2S awareness', 'hazard identification', 'heat load calculation', 'high pot testing', 'HV testing', 'hydro testing', 
    'Induction Motors', 'insulation resistance test', 'isometric drawing', 'JSA', 'keyword optimization', 'lighting layout', 'Load Runner', 'lumen testing',
    'maintenance', 'manpower planning', 'manual handling', 'microcontroller', 'microprocessor', 'Microsoft Office', 'operation & maintenance', 
    'open loop checking', 'password management', 'permit to work system', 'site supervision', 'testing', 'tool box talk', 'web/WWW', 'Win XP', 
    'Win8', 'WordPress plugin', 'working at height', 'Xilinx',



    # 500 - 1000
    'PFD', 'PH meter', 'pressure gauge', 'pressure testing', 'pressure transmitter', 'Pro/E', 'ProE 2000i', 'pump head calculation', 'PVSyst', 'R software', 
    'relays', 'Sand test', 'scaffold safety', 'scaffolding', 'smoke testing', 'SMTP', 'specific gravity', 'STAAD Pro', 'star delta', 'stock ledger', 
    'subdivision distribution board', 'super tension cable', 'Tally', 'Tally ERP', 'TeamViewer', 'telephone systems', 'testing & commissioning', 'total station', 
    'transmission & distribution', 'ultrasonic testing', 'web design', 'Windows', 'WinCC', 'working at height', '2D drawing', '3-phase power systems', '5S', '7QC tools', 
    'Arc welding', '8051 microcontroller', 'AutoCAD', 'CAD tools', 'cash flow analysis', 'CCTV/IP camera setup', 'control systems', 'electrical design', 'estimation', 
    'HVAC systems', 'inspection', 'instrumentation', 'maintenance', 'mechanical work', 'microcontroller', 'panel modifications', 'preventive maintenance', 
    'quality control', 'scaffolding safety', 'testing', 'valves', 'welding',


    # 1000 - 1500
    'ARPA', 'artificial intelligence', 'asset management', 'audit', 'AutoCAD', 'AutoCAD 2D', 'AutoPLANT', 'AviCAD', 'AWG', 'AWT', 'balance sheet', 'bank reconciliation', 
    'bar bending schedule', 'Barracuda firewall', 'basic firefighting', 'BMS', 'BMS & HVAC basics', 'board testing', 'bookkeeping', 'books of accounts', 'boundary survey', 
    'breakdown maintenance', 'bug tracking', 'Bugzilla', 'building construction', 'building wiring', 'business analysis', 'business development', 'C', 'C++', 'CAD', 
    'CADWorx', 'CAE software', 'Caesar II', 'calculation', 'cable laying', 'cable sizing', 'cable testing', 'CAD drafting', 'cash flow analysis', 'CCTV', 'control systems', 
    'CPR', 'CRM', 'data analysis', 'electrical engineering', 'estimation', 'firefighting', 'HVAC', 'inspection', 'instrumentation', 'IT systems', 'maintenance', 
    'mechanical engineering', 'PLC', 'SCADA', 'testing', 'welding',

    # 1500 - 2000
    'Capacitance measurement', 'capital asset identification', 'carbon dioxide detection', 'card testing', 'cardio pulmonary rehabilitation', 'cash book', 'cash handling', 
    'CATIA V5', 'CB sizing calculation', 'centrifugal pumps', 'centroid calculation', 'change management', 'checkpoint firewalls', 'circuit design', 'circuit theory', 
    'Citect SCADA', 'civil drafting', 'civil engineering', 'claim processing', 'client billing', 'client handling', 'CNC machines', 'CNC programming', 'cold calling', 
    'collection of payments', 'commercial evaluation', 'commissioning', 'communication skills', 'component testing', 'compressor maintenance', 'computer literacy', 
    'concrete batching', 'concrete casting', 'concrete mix design', 'conductivity measurement', 'confined space entry', 'construction management', 'construction quality management', 
    'control systems', 'control panels', 'control circuits', 'control logix', 'coordination', 'corporate finance', 'cost accounting', 'cost control', 'cost reduction', 'CPR', 
    'credit appraisal', 'credit control', 'Creo/Pro-E', 'crimping', 'critical task analysis',

    # 2000 - 2500
    'cross cut adhesion testing', 'cross selling', 'CRUD', 'cryptanalysis', 'CT analyzer', 'CT/PT sizing', 'current transformer sizing', 'Curve Tracer', 
    'customer relationship management', 'Cyberoam firewall', 'Cygwin', 'data validation', 'DC sheath test', 'DC winding resistance', 'DCS FAT', 'debridement', 
    'debtors & creditors', 'design & development', 'design drafting', 'detailed drawings', 'develop test cases', 'device net', 'dew point measurement', 
    'DG set maintenance', 'DG sizing calculation', 'DGPS operations', 'diagnosing faults', 'DIALux', 'digital marketing', 'direct torque control', 'disaster management', 
    'dispatch management', 'distribution planning', 'document control', 'documentation', 'DOL starter', 'Dropbox', 'DSP', 'DTMF decoder', 'duct design', 'duct leak test', 
    'dumpy level', 'e-commerce', 'e-mail', 'E-plan', 'e-recruitment', 'E-survey', 'Eagle Layout Editor', 'earth resistance testing', 'earthworks', 'earthquake resistant design', 
    'ECAD', 'EBITDA', 'EHS awareness', 'electrical control circuits', 'electrical distribution', 'electrical installation', 'electrical load calculation', 'electrical maintenance', 
    'electrical systems maintenance', 'electromagnetism', 'electronics design', 'embedded C', 'embedded systems', 'emergency management',

    # 2500 - 3000
    'emulation', 'cryptography', 'energy management', 'EPABX systems', 'equipment layout', 'equipment selection', 'erection & commissioning', 'ERP modules', 
    'error handling', 'Ethernet IP', 'ETL', 'evacuation planning', 'excavation', 'Excel', 'Exchange Server', 'export/import', 'extraction & reporting', 'fabrication', 
    'fault current analysis', 'FAT analysis', 'FCAW welding', 'feedback loops', 'feeder maintenance', 'field calibration', 'field instrumentation', 'FileZilla', 
    'filter press operation', 'finalization of accounts', 'financial analysis', 'fire detection systems', 'fire fighting design', 'fire prevention', 'firewall', 
    'firmware upgrade', 'first aid', 'fishbone analysis', 'fixed asset management', 'flow measurement', 'Fluke instruments', 'foundation work', 'front office management', 
    'functionality testing', 'fund flow analysis', 'gas cutting', 'gas welding', 'gate valves', 'GD&T', 'general ledger', 'generating leads', 'geotechnical analysis', 
    'GMAW welding', 'GNU', 'Google Maps', 'gloss measurement',

    # 3000 - 3500
    'AutoCAD', 'PlanSwift', 'GRN (Goods Receipt Note)', 'graphic software', 'grouting', 'GSM', 'GST', 'heat load calculation', 'heat transfer', 'heat treatment', 
    'hardness testing', 'hardware testing', 'hash tables', 'hazard management', 'HAZOP', 'HAP software', 'HVAC load calculation', 'HVAC systems', 'handset repair', 
    'HMI (human machine interface)', 'humidity measurement', 'hydraulic systems', 'pneumatic systems', 'hydro testing', 'HPLC', 'HR management', 'HSE management', 
    'HTML/HTTPS', 'hypercard', 'IBS testing', 'ICT support', 'identifying hazards', 'impact testing', 'import/export', 'incident management', 'incoming inspection', 
    'industrial maintenance', 'inergen systems', 'information systems', 'inspection', 'installation & commissioning', 'instrumentation', 'insulation resistance testing', 
    'integrated management systems', 'interpersonal skills', 'inventory control', 'in-process quality control', 'innovation', 'process improvement',

    # 3500 - 4000
    'international marketing', 'internet concepts', 'internet programming', 'inventory management', 'inverter systems', 'iptables', 'IPTV', 'isometric drawing', 
    'IT desktop support', 'IT maintenance', 'ITIL', 'Jasper Reports', 'JQuery Mobile', 'k-means clustering', 'Kaizen', 'Karl Fischer titration', 'Key Account Management', 
    'KNX systems', 'LAN networking', 'land surveying', 'lathe operations', 'layout design', 'leak testing', 'lean methodology', 'ledger management', 'legal documentation', 
    'letter of credit', 'level transmitter', 'lighting load calculation', 'liquid penetrant testing', 'load calculation', 'lockout tagout (LOTO)', 'logistics management', 
    'loop checking', 'lubrication systems', 'lux calculation', 'magnetic particle testing', 'mail server management', 'maintenance & troubleshooting', 'maintenance management', 
    'MIS reporting', 'motor maintenance', 'machine handling', 'machine maintenance',

    # 4000 - 4500
    'audit', 'makefile', 'MAMP', 'management accounting', 'management information systems', 'manual testing', 'MapInfo', 'market analysis', 'material handling', 'material planning', 
    'material take-off', 'Mathematica', 'Maximo', 'McQuay DuctSizer', 'mechanical design', 'MEP design', 'Mercury Quality Centre', 'Metasys', 'meter testing', 'microgrid', 
    'MicroStation', 'microbial testing', 'microcontroller 8051', 'microcontroller AT89C51', 'MicroLogix', 'microprocessor 8085', 'MIG welding', 'TIG welding', 'MikroTik', 
    'MIS audits', 'mixing and fluid testing', 'ModelSim', 'module testing', 'monitoring and inspection', 'MPLAB IDE', 'MPPT controller', 'MS Office', 'MS PowerPoint', 
    'MS Publisher', 'MS-DOS', 'VLOOKUP', 'MTBF', 'MTTR', 'multimedia systems', 'multivariable statistics', 'MVC', 'Myrinet', 'numerical relay testing', 'Navicat', 'Navisworks', 
    'NDT', 'network systems', 'NGT/NGR sizing', 'nitration', 'NNTP', 'non-destructive testing', 'normal and emergency lighting', 'motor maintenance', 'maintenance & repair', 
    'maintenance troubleshooting', 'maintenance planning', 'cash handling', 'ledger scrutiny', 'invoice processing', 'banking operations', 'record keeping', 'safety management', 
    'HSE management', 'site monitoring', 'installation & commissioning', 'inspection', 'mechanical works', 'electrical maintenance', 'cable laying', 'lubrication',

    # 4500 - 5000
    'Novec 1230 systems', 'numerical analysis', 'object oriented programming', 'object repository', 'object spy', 'OFC checking', 'OHSAS', 'oil handling', 'oil in water analysis', 
    'Omron HMI', 'OpenNMS', 'operating systems', 'operation & maintenance', 'Oracle 10g', 'Oracle 11g', 'OrCAD', 'order to cash', 'osCommerce', 'Outlook Express', 'overhead tank sizing', 
    'P&ID', 'P&ID mark-up', 'package AC systems', 'palletizing', 'Pareto analysis', 'particle count testing', 'password cracking', 'payroll management', 'PCB design', 'PCC & MCC panels', 
    'PCS7', 'PDF tools', 'Peachtree accounting', 'penetrant testing', 'pencil hardness testing', 'performance analysis', 'period close analysis', 'petroleum exploration', 'PF Sense firewall', 
    'phase sequencer', 'physical metallurgy', 'physiotherapy techniques', 'PIC microcontrollers', 'PID isometric drawings', 'pile foundation design', 'piping design', 'piping layout', 
    'pivot tables', 'planning & scheduling', 'plant downtime calculation', 'plasma cutting', 'plastering', 'PLC programming', 'plot plans', 'plumbing systems', 'point of sale (POS)', 
    'poka-yoke', 'polymer synthesis', 'POP3', 'poster design', 'postoperative pain management', 'power electronics', 'power plant management', 'power transmission',

    # 5000 - 5500
    'PPAP', 'PPC scheduling', 'practical accounting', 'pre-weld inspection', 'precast concrete', 'predictive maintenance', 'preventive maintenance', 'BOQ preparation', 
    'tender preparation', 'bill preparation', 'as-built drawings', 'bar bending schedules', 'Primavera P6', 'Pro/E', 'problem solving', 'process planning', 'process safety', 
    'process control', 'invoice processing', 'procurement', 'production planning', 'production management', 'quality control', 'quality assurance', 'quality systems', 'Profibus', 
    'profit & loss accounting', 'programming', 'proofreading', 'prototype development', 'Proteus', 'PScad', 'psychrometric chart', 'pump head calculation', 'PV Elite', 'PV systems', 
    'QC inspection', 'QA/QC', 'quantity surveying', 'QuickBooks', 'quotation management', 'radiographic testing', 'rainwater harvesting', 'RCCB', 'rebar fixing', 'reconciliation of accounts', 
    'recruitment', 'reinforcement checking', 'relationship management', 'Relux lighting software', 'repair & maintenance', 'report writing', 'research', 'resource management', 'risk management', 
    'root cause analysis', 'reading blueprints',

    # 5500 - 6000
    'risk assessment', 'rigging operations', 'rigging safety', 'rock testing', 'RSLogix', 'RSView32', 'rsync', 'RTOS', 'RTGS', 'safety management', 'sales planning', 'sales & marketing', 
    'sample preparation', 'sanity testing', 'SAP BI', 'SAP FI', 'SCADA', 'scaffolding', 'scaffolding safety', 'schedule preparation', 'scheme checking', 'Scilab', 'scope meter', 'screening candidates', 
    'scrutiny', 'sealing work', 'seaming machines', 'secondary surveys', 'selection & power calculation', 'self compacting concrete', 'selling techniques', 'server 2008', 'service & maintenance', 
    'sewage treatment plant', 'Sewercad', 'sheet piling', 'sheet metal fitting', 'shift planning', 'short circuit calculation', 'shutdown planning', 'Silverlight', 'site evaluation', 'site execution', 
    'site inspection', 'site mobilization', 'SketchUp', 'slab checking', 'sling psychrometer', 'slitting', 'smart grid', 'smart meters', 'SmartPlant 3D', 'SmartPlant P&ID', 'SMAW welding', 'SMPS', 'SMT', 
    'software development', 'software testing', 'software testing life cycle', 'soil impact test', 'solar energy', 'solar panel installation', 'solar thermal design', 'SolidWorks', 'Solid Edge', 
    'SonicWall firewall', 'SOP records', 'spatial indexing', 'speed control', 'spectrophotometer', 'spin coating', 'spinal surgery techniques', 'spot welding', 'spray nozzle systems', 
    'sprint burndown charts', 'statistical quality control', 'SQC methods',

    # 6000 - 6500
    'SRAM', 'SRS', 'STAAD Pro', 'stabilization of patients', 'stage-wise testing', 'stakeout method', 'stamp concrete', 'star-delta starter', 'static pressure calculation', 'static regain method', 
    'statutory audit', 'statutory reporting', 'steam blowing', 'steam turbine', 'steel erection', 'steel fixing', 'steganography', 'stock audit', 'stock management', 'stock taking', 
    'stress analysis software', 'structural modelling', 'student supervision', 'Studio 5000', 'subnetting', 'SunOS', 'supervision & control', 'supplier quality management', 'surface preparation', 
    'survey software', 'symmetric key cryptography', 'Systat', 'system integration', 'system testing', 'Tally ERP', 'team building', 'technical evaluation', 'technical support', 'telecalling', 
    'telecom systems handling', 'Telerik', 'temperature transmitter', 'temperature & RH testing', 'TEMS', 'tendon suturing', 'tensile testing', 'testing & commissioning', 'earth resistance testing', 
    'thermocouple', 'RTD', 'thermodynamics', 'thermal cycling', 'three-phase induction motor starter', 'theory of machines', 'tool box talks', 'topographic survey', 'torque testing', 
    'total quality management', 'total station', 'total sulfur analyzer',

    # 6500 - 7000
    'TIA Portal / Totally Integrated Automation', 'tower crane erection', 'tower erection', 'tower maintenance', 'traceability matrix', 'traffic safety', 'first aid', 'training coordination', 
    'training needs analysis', 'transformer sizing', 'transmission systems', 'traverse survey', 'trend and variance analysis', 'Trend Micro Antivirus', 'trial balance', 'troubleshooting', 
    'ultrasonic testing', 'UG NX', 'Unigraphics NX', 'unit testing', 'UPS systems', 'utility and plant maintenance', 'V-Ray', 'valve systems', 'variance analysis', 'vector network analysis', 
    'vendor management', 'VFD', 'visual and dimensional inspection', 'Visual Basic', 'Visual Studio', 'VLSI', 'VNC', 'voltage drop calculation', 'voltage regulation', 'VRML', 'VRF/VRV systems', 
    'WAMP', 'WAN', 'waste management', 'effluent management', 'waterproofing', 'WaterCAD', 'WBS planning', 'web testing', 'website testing', 'welding', 'cutting', 'well testing', 'white box testing', 
    'Wi-Fi / radio systems', 'WinBox', 'WinRAR', 'WinSCP', 'wire bonding', 'witness pressure testing', 'WMM checking', 'workshop management', 'WPS', 'WSUS', 'Xcode', 'Xilinx ISE', 'yield calculation', 'YOLO CNN',
}

ENTITY_GROUPS = {
    "skills": SKILL_LABELS,
    "person_name": ["person name (individual human name)"],
    "job_title": ["job title", "professional role", "occupation", "position held in a company"],
    "company": ["company", "organization", "employer", "workplace company"],
    "university": ["university", "higher education institution", "college"],
    "degree": ["degree"],
    "address": ["full address", "residential address", "home address"],}

ALL_LABELS = list(dict.fromkeys(
    label
    for labels in ENTITY_GROUPS.values()
    for label in labels
))

ACADEMIC_KEYWORDS = {
    "mestrado",
    "licenciatura",
    "doutoramento",
    "doutorado",
    "bachelor",
    "master",
    "phd",
    "degree",
    "engenharia",
    "engineering",
    "industrial",}

INVALID_ENTITIES = {
    "person_name": {
        "person","personal","personal data","my responsibility","smoking","no smoking","role","designation","good","well","bad","real","job description","job prospectus","fast","hand","gay","certificate","last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background","academic","background","education", "experience", "hobbies"},
    "job_title": {
        "person","personal","personal data","my responsibility","smoking","no smoking","head","cool","intermediate","basic","high","advanced","role","designation","good","well","bad","real","job description","job prospectus","fast","hand","gay","first name", "certificate","last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background", "background","education", "experience", "hobbies", "kangaroo",
        "Area of interest",
        "vista",
        "career",
        "Technical Qualification",
        "civil",
        "dir",
        "Job Responsibilities",
        "Married",
        "challenging position",
        "Executive",
        "Trainee",
        "abilities",
        "cricket",
        "Listening music",
        "QTP",
        "Basics",
        "e-mail",
        "potential",
        "accounts",
        "BOQ",
        "Areas of Interest",
        "DWG",
        "Listening to music",
        "craft persons",
        "job satisfaction",
        "laborers",
        "makeup",
        "Occupation",
        "PCRs",
        "tenants",
        "Vogue",
        "expert",
        "Flaunt",
        "Speak",
        "Sr.",
        "swimming",
        "Glamour",
        "HOD",
        "Junior resident",
        "Linguistic Proficiency",
        "Playing cricket",
        "Write",
        "Askim Online English Tutorial",
        "Badminton",
        "Declaration",
        "Elle",
        "HAP",
        "member",
        "Personal Strength",
        "post",
        "Signature",
        "Singing",
        "Single",
        "Supervisors",
        "Turkish students",
        "American Idol",
        "analyze",
        "attained maximum profit",
        "CEO",
        "CG Lux",
        "Corel draw",
        "dancing",
        "DCA",
        "erection",
        "Field of Interest",
        "Firefighting system",
        "First Class",
        "General",
        "incumbent",
        "KEIL",
        "Major",
        "MEP field",
        "NDT",
        "O & M",
        "Plan",
        "profession",
        "R & D",
        "Runway",
        "Site Incharge",
        "SLD",
        "Specialist",
        "Supervise",
        "talents",
        "Topic",
        "Training & Development",
        "VHDL",
        "ACB",
        "Apprentice",
        "Basic Core",
        "BasicKnowledgeofComputer",
        "BOSS",
        "C & I",
        "casting",
        "challenging career",
        "Challenging job",
        "Dealers",
        "Directors",
        "Event organizer",
        "FirstAidfromRedCross",
        "Graduation",
        "hair",
        "Hairstylist & Makeup Artist",
        "Incharge",
        "ISI testing of MCBs",
        "key player",
        "KEY Responsibilities",
        "key role",
        "Lead",
        "Matriculation",
        "MIG",
        "patients",
        "PDMS",
        "Personal Information",
        "personal trait",
        "Personality Traits",
        "Phillips Driver",
        "Position Held",
        "Predictive",
        "Preparation of Bills of documents",
        "Preparation of programme of works",
        "Preparation of Tender documents",
        "Preparation of variation orders",
        "Professional environment",
        "Resident","student", "students",
        "Subject",
        "symbiotic relationship", "Teen Vogue", "Testing & Quality Control Engineer Representative",
        "Tiles Mason", "TRT testing","value addition",
        "VFD", "Work Order",
        "workers",
        "Accounting Package",
        "Aim",  "ARC",
        "Architectural",
        "asst",
        "Austin Powers","Basics of launguages",
        "Beginner","Bluemix","Brake test","Branch","calculation",
        "Captain",
        "challenging and responsible position",
        "Checkpoint",
        "civil helper",
        "civil Maintenance",
        "Communicator",
        "Confined Space",
        "Cost ascertainment",
        "critical thinker",
        "cross-team",
        "department head",
        "Dept.",
        "development department",
        "DTP",
        "ECE",
        "EFA",
        "Effective communicator",
        "Employee",
        "Engg",
        "EViews",
        "exel",
        "Familiarity",
        "field",
        "Friendly",
        "Fundamentals",
        "Global Warming",
        "Graduate Apprentice Trainee",
        "high-level position",
        "HSE",
        "in charge",
        "in depth knowledge",
        "In Style",
        "install",
        "Interest",
        "interface",
        "ISTP",
        "key position",
        "Key Result Areas",
        "Knowledge in I nstallation of precast hollow core roof slabs",
        "LA Confidential",
        "ledger",
        "level of study",
        "Life member",
        "Manual",
        "Marie Claire",
        "MCCB",
        "MDB",
        "Method",
        "Micrometer",
        "MIGO",
        "My Role",
        "Obtaining DEWA engineer",
        "P&M",
        "P.Point",
        "packing issuer",
        "PCC",
        "permit to work",
        "Personal Interests",
        "Personal Qualities",
        "positive attitude",
        "potentials",
        "preventive",
        "preventive & maintenance",
        "Reading novels",
        "Responsible for attending the breakdown maintenance",
        "RTD",
        "S oil density testing",
        "Site Diary",
        "SMDB",
        "Software Skils",
        "soun d understanding",
        "Specialisation",
        "Spine Surgery",
        "ss7",
        "staffs",
        "Structural",
        "suitable position",
        "suitable post",
        "Surfing",
        "Technical Abilties",
        "technical inputs",
        "technical issues",
        "Technical Qualifications",
        "TIG",
        "TORA",
        "towards life",
        "Unmarried"},
    "skills": {
        "person","personal","personal data","my responsibility","smoking","no smoking","head","cool","intermediate","basic","high","advanced","role","designation","good","well","bad","real","job description","job prospectus","fast","hand","gay","first name", "certificate", "last name", "surname","skill", "skills", "soft skill", "hard skill","job", "title", "position", "suffix", "prefix", "academic background","academic","background","education", "experience", "hobbies", "c1", "first certificate",
        "Area of interest",
        "vista",
        "career",
        "Technical Qualification",
        "civil",
        "dir",
        "Job Responsibilities",
        "Married",
        "challenging position",
        "Executive",
        "Trainee",
        "abilities",
        "cricket",
        "Listening music",
        "QTP",
        "Basics",
        "e-mail",
        "potential",
        "accounts",
        "BOQ",
        "Areas of Interest",
        "DWG",
        "Listening to music",
        "craft persons",
        "job satisfaction",
        "laborers",
        "makeup",
        "Occupation",
        "PCRs",
        "tenants",
        "Vogue",
        "expert",
        "Flaunt",
        "Speak",
        "Sr.",
        "swimming",
        "Glamour",
        "HOD",
        "Junior resident",
        "Linguistic Proficiency",
        "Playing cricket",
        "Write",
        "Askim Online English Tutorial",
        "Badminton",
        "Declaration",
        "Elle",
        "HAP",
        "member",
        "Personal Strength",
        "post",
        "Signature",
        "Singing",
        "Single",
        "Supervisors",
        "Turkish students",
        "American Idol",
        "analyze",
        "attained maximum profit",
        "CEO",
        "CG Lux",
        "Corel draw",
        "dancing",
        "DCA",
        "erection",
        "Field of Interest",
        "Firefighting system",
        "First Class",
        "General",
        "incumbent",
        "KEIL",
        "Major",
        "MEP field",
        "NDT",
        "O & M",
        "Plan",
        "profession",
        "R & D",
        "Runway",
        "Site Incharge",
        "SLD",
        "Specialist",
        "Supervise",
        "talents",
        "Topic",
        "Training & Development",
        "VHDL",
        "ACB",
        "Apprentice",
        "Basic Core",
        "BasicKnowledgeofComputer",
        "BOSS",
        "C & I",
        "casting",
        "challenging career",
        "Challenging job",
        "Dealers",
        "Directors",
        "Event organizer",
        "FirstAidfromRedCross",
        "Graduation",
        "hair",
        "Hairstylist & Makeup Artist",
        "Incharge",
        "ISI testing of MCBs",
        "key player",
        "KEY Responsibilities",
        "key role",
        "Lead",
        "Matriculation",
        "MIG",
        "patients",
        "PDMS",
        "Personal Information",
        "personal trait",
        "Personality Traits",
        "Phillips Driver",
        "Position Held",
        "Predictive",
        "Preparation of Bills of documents",
        "Preparation of programme of works",
        "Preparation of Tender documents",
        "Preparation of variation orders",
        "Professional environment",
        "Resident",
        "student",
        "students",
        "Subject",
        "symbiotic relationship",
        "Teen Vogue",
        "Testing & Quality Control Engineer Representative",
        "Tiles Mason",
        "TRT testing",
        "value addition",
        "VFD",
        "Work Order",
        "workers",
        "Accounting Package",
        "Aim",
        "ARC",
        "Architectural",
        "asst",
        "Austin Powers",
        "Basics of launguages",
        "Beginner",
        "Bluemix",
        "Brake test",
        "Branch", "calculation",
        "challenging and responsible position","Checkpoint","Communicator",
        "Confined Space","Cost ascertainment","Dept.","development department",
        "DTP","ECE", "EFA","Employee",
        "Engg", "EViews",
        "exel","Familiarity",
        "field","Friendly","Fundamentals",
        "Global Warming", 
        "high-level position","HSE",
        "in charge","in depth knowledge","In Style",
        "install", "Interest",
        "interface", "ISTP",
        "key position",
        "Key Result Areas",
        "Knowledge in I nstallation of precast hollow core roof slabs",
        "LA Confidential",
        "ledger",
        "level of study",
        "Life member",
        "Manual",
        "Marie Claire",
        "MCCB",
        "MDB",
        "Method",
        "Micrometer",
        "MIGO",
        "My Role",
        "Obtaining DEWA engineer",
        "P&M",
        "P.Point",
        "packing issuer",
        "PCC",
        "permit to work",
        "Personal Interests",
        "Personal Qualities",
        "positive attitude",
        "potentials",
        "preventive",
        "preventive & maintenance",
        "Reading novels",
        "Responsible for attending the breakdown maintenance",
        "RTD",
        "S oil density testing",
        "Site Diary",
        "SMDB",
        "Software Skils",
        "soun d understanding",
        "Specialisation",
        "Spine Surgery",
        "ss7",
        "staffs",
        "Structural",
        "suitable position",
        "suitable post",
        "Surfing",
        "Technical Abilties",
        "technical inputs",
        "technical issues",
        "Technical Qualifications",
        "TIG",
        "TORA",
        "towards life",
        "Unmarried"},
    "degree": {"degree", "education", "diploma", "certificate", "executive education"},
    "company": {"company", "organization", "linkedin", "github", "linkdin", "microsoft"},
    "address": {"address", "location"}}


GENERIC_INVALID = {
    "person",
    "certificate",
    "education",
    "experience",
    "skills",
    "skill",
    "hobbies",
}

POSTAL_PATTERN = re.compile(r"\b\d{4,6}[- ]?\d{2,4}\b")
NUMBER_PATTERN = re.compile(r"\b(nº|no\.?|number)?\s?\d+\b", re.IGNORECASE)

NAME_TOKEN_PATTERN = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]+$")


# ----------------------------------------------------------
# Canonical normalization for controlled vocabularies
# ----------------------------------------------------------

_CAMEL_BOUNDARY_REGEX = re.compile(r"(?<=[a-zà-öø-ÿ])(?=[A-ZÀ-ÖØ-Þ])")
_MULTI_SEP_REGEX = re.compile(r"\s*-\s*|\s+")


def _split_camel_case(text: str) -> str:
    """
    MicrosoftOffice -> Microsoft Office
    powerBI -> power BI
    """
    if not text:
        return text
    return _CAMEL_BOUNDARY_REGEX.sub(" ", text)


def _canonical_display_form(text: str) -> str:
    """
    Forma de output:
    - separa camel case
    - troca hífens por espaço
    - colapsa espaços
    """
    if not text:
        return ""

    t = _split_camel_case(text)
    t = re.sub(r"\s*-\s*", " ", t)
    t = " ".join(t.split()).strip()

    return t


def _canonical_choice_score(text: str) -> tuple:
    """
    Score para escolher a melhor forma canónica quando há duplicados
    no vocabulário com a mesma normalização.
    Preferências:
    1) menos hífens
    2) mais espaços legíveis
    3) mais maiúsculas úteis
    4) texto mais comprido
    """
    if not text:
        return (10**9, -10**9, -10**9, -10**9)

    hyphens = text.count("-")
    spaces = text.count(" ")
    uppers = sum(1 for ch in text if ch.isupper())
    length = len(text)

    return (hyphens, -spaces, -uppers, -length)


def _normalize_if_in_vocab(text: str, vocab_index: Dict[str, str]) -> Optional[str]:
    """
    Só normaliza se a forma normalizada existir no vocabulário permitido.
    """
    if not text:
        return None

    norm = _normalize_vocab_surface(text)
    if not norm:
        return None

    return vocab_index.get(norm)



def _normalize_vocab_surface(text: str) -> str:
    """
    Normalização leve para comparação contra vocabulários controlados.

    Exemplos:
        Microsoft Office     -> microsoft office
        Microsoft-Office     -> microsoft office
        Microsoft - Office   -> microsoft office
        MicrosoftOffice      -> microsoft office
        ms-office            -> ms office
    """
    if not text:
        return ""

    t = text.strip()

    # 1) split camelCase / PascalCase
    t = _split_camel_case(t)

    # 2) normalizar hífen/espaços para um único espaço
    t = _MULTI_SEP_REGEX.sub(" ", t)

    # 3) limpar espaços repetidos
    t = " ".join(t.split()).strip().lower()

    return t


def _build_flexible_vocab_regex(term: str) -> re.Pattern[str]:
    """
    Gera regex para apanhar:
      - Microsoft Office
      - Microsoft-Office
      - Microsoft- Office
      - Microsoft -Office
      - Microsoft - Office
      - MicrosoftOffice
    """
    normalized = _normalize_vocab_surface(term)
    tokens = normalized.split()

    if not tokens:
        return re.compile(r"(?!x)x")

    if len(tokens) == 1:
        token = re.escape(tokens[0])
        return re.compile(rf"(?<!\w){token}(?!\w)", re.IGNORECASE)

    connector = r"(?:\s*-\s*|\s+|)"
    pattern = r"(?<!\w)" + connector.join(re.escape(tok) for tok in tokens) + r"(?!\w)"
    return re.compile(pattern, re.IGNORECASE)


def _build_vocab_index(vocab: Set[str]) -> Dict[str, str]:
    """
    Cria um índice:
        forma_normalizada -> melhor forma canónica de output
    """
    index: Dict[str, str] = {}

    for item in vocab:
        if not item or not str(item).strip():
            continue

        raw = str(item).strip()
        norm = _normalize_vocab_surface(raw)
        display = _canonical_display_form(raw)

        if not norm:
            continue

        current = index.get(norm)
        if current is None:
            index[norm] = display
        else:
            if _canonical_choice_score(display) < _canonical_choice_score(current):
                index[norm] = display

    return index



# ----------------------------------------------------------
# Normalize labels, regex, etc
# ----------------------------------------------------------


# Normalize language list once
LANGUAGES_SET = {lang.lower().strip() for lang in LANGUAGES}


JOB_TITLE_NORMALIZATION_INDEX = _build_vocab_index(set(SIMPLE_JOB_TITLES))
SOFTWARE_NORMALIZATION_INDEX = _build_vocab_index(set(SOFTWARE_AS_SKILL))
PROGRAMMING_NORMALIZATION_INDEX = _build_vocab_index(set(PROGRAMMING_LIBRARIES))
GENERAL_SKILLS_NORMALIZATION_INDEX = _build_vocab_index(set(GENERAL_SKILLS))


JOB_TITLE_ITEMS_SORTED = sorted(
    JOB_TITLE_NORMALIZATION_INDEX.items(),
    key=lambda x: len(x[0]),
    reverse=True
)

SOFTWARE_ITEMS_SORTED = sorted(
    SOFTWARE_NORMALIZATION_INDEX.items(),
    key=lambda x: len(x[0]),
    reverse=True
)

PROGRAMMING_ITEMS_SORTED = sorted(
    PROGRAMMING_NORMALIZATION_INDEX.items(),
    key=lambda x: len(x[0]),
    reverse=True
)

GENERAL_SKILLS_ITEMS_SORTED = sorted(
    GENERAL_SKILLS_NORMALIZATION_INDEX.items(),
    key=lambda x: len(x[0]),
    reverse=True
)

JOB_TITLE_REGEX_PATTERNS = [
    (_build_flexible_vocab_regex(canonical_display), canonical_display)
    for _, canonical_display in JOB_TITLE_ITEMS_SORTED
]

SOFTWARE_REGEX_PATTERNS = [
    (_build_flexible_vocab_regex(canonical_display), canonical_display)
    for _, canonical_display in SOFTWARE_ITEMS_SORTED
]

PROGRAMMING_REGEX_PATTERNS = [
    (_build_flexible_vocab_regex(canonical_display), canonical_display)
    for _, canonical_display in PROGRAMMING_ITEMS_SORTED
]

GENERAL_SKILLS_REGEX_PATTERNS = [
    (_build_flexible_vocab_regex(canonical_display), canonical_display)
    for _, canonical_display in GENERAL_SKILLS_ITEMS_SORTED
]




# ----------------------------------------------------------
# Langue Filters
# ----------------------------------------------------------

def _is_language_entity(text: str) -> bool:
    """
    True se a entidade corresponder a uma língua conhecida.
    """
    if not text:
        return False

    t = " ".join(text.lower().strip().split())

    if t in LANGUAGES_SET:
        return True

    return False


# ----------------------------------------------------------
# Person Name Filters and Helpers
# ----------------------------------------------------------

def _contains_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text or "")


def _clean_single_person_name(text: str) -> Optional[str]:
    """
    Limpa uma única entidade person_name usando a lógica já existente.
    """
    cleaned = _clean_person_name([text])
    return cleaned[0] if cleaned else None


def _choose_best_person_name(
    candidates: List[Dict[str, Any]],
    score_margin: float = 0.05,
) -> List[str]:
    """
    Escolhe apenas 1 nome.

    Regras:
    1. Maior score
    2. Se diferença entre top1 e top2 < score_margin,
       escolher o primeiro candidato sem números
    3. Se não existir, usar o top1
    """
    if not candidates:
        return []

    # Deduplicar por texto limpo, mantendo:
    # - maior score
    # - menor posição (mais cedo no texto) em caso de empate
    best_by_text: Dict[str, Dict[str, Any]] = {}

    for c in candidates:
        raw_text = str(c.get("text", "")).strip()
        if not raw_text:
            continue

        cleaned = _clean_single_person_name(raw_text)
        if not cleaned:
            continue

        score = float(c.get("score", 0.0))
        start = int(c.get("start", 10**9))

        key = cleaned.lower()
        current = best_by_text.get(key)

        candidate = {
            "text": cleaned,
            "score": score,
            "start": start,
        }

        if current is None:
            best_by_text[key] = candidate
            continue

        if candidate["score"] > current["score"]:
            best_by_text[key] = candidate
        elif candidate["score"] == current["score"] and candidate["start"] < current["start"]:
            best_by_text[key] = candidate

    final_candidates = list(best_by_text.values())
    if not final_candidates:
        return []

    # Ordenar por:
    # 1. score desc
    # 2. posição asc
    final_candidates.sort(key=lambda x: (-x["score"], x["start"]))

    top1 = final_candidates[0]

    if len(final_candidates) == 1:
        return [top1["text"]]

    top2 = final_candidates[1]
    score_diff = top1["score"] - top2["score"]

    # Se os scores forem muito próximos, escolher o primeiro sem números
    if score_diff < score_margin:
        first_without_digits = min(
            (c for c in final_candidates if not _contains_digits(c["text"])),
            key=lambda x: x["start"],
            default=None
        )
        if first_without_digits is not None:
            return [first_without_digits["text"]]

    return [top1["text"]]

def _extract_best_person_name_from_top(
    text: str,
    model: Optional[GLiNER] = None,
    threshold: float = 0.15,
    top_window: int = 450,
) -> List[str]:
    """
    Extrai apenas o melhor person_name da zona de topo do CV.
    """
    if not text or not text.strip():
        return []

    top_text = text[:top_window]

    entities = extract_entities_from_text(
        top_text,
        model=model,
        threshold=threshold,
        debug_save_json=False,
        debug_output_path=None,
        use_regex_fallbacks=False,
        use_job_title_regex_fallback=False,
        use_skill_regex_fallback=False,
        use_software_regex_fallback=False,
        use_programming_regex_fallback=False,
        use_general_skills_regex_fallback=False,
    )

    names = entities.get("person_name", []) or []

    heuristic_name = extract_top_name_candidate(text)

    if not names:
        return [heuristic_name] if heuristic_name else []

    if heuristic_name:
        first_found = names[0]

        if len(first_found.split()) == 1 and len(heuristic_name.split()) >= 2:
            first_token = first_found.lower()
            heuristic_tokens = [t.lower() for t in heuristic_name.split()]

            if first_token in heuristic_tokens:
                return [heuristic_name]

    return [names[0]]


# ----------------------------------------------------------
# Regex functions (fallback for gliner)
# ----------------------------------------------------------
def _extract_from_precompiled_patterns(
    text: str,
    patterns: List[Tuple[re.Pattern[str], str]],
    validator: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    if not text:
        return []

    results: List[str] = []
    seen: Set[str] = set()

    for rx, canonical_display in patterns:
        if rx.search(text):
            key = canonical_display.lower()

            if key in seen:
                continue

            if validator is not None and not validator(canonical_display):
                continue

            seen.add(key)
            results.append(canonical_display)

    return results

def _extract_software_as_skill_from_text(text: str) -> List[str]:
    return _extract_from_precompiled_patterns(
        text,
        SOFTWARE_REGEX_PATTERNS,
        validator=_is_valid_skill,
    )

def _extract_programming_libraries_from_text(text: str) -> List[str]:
    return _extract_from_precompiled_patterns(
        text,
        PROGRAMMING_REGEX_PATTERNS,
    )

def _extract_general_skills_from_text(text: str) -> List[str]:
    return _extract_from_precompiled_patterns(
        text,
        GENERAL_SKILLS_REGEX_PATTERNS,
    )



# ----------------------------------------------------------
# Gliner functions
# ----------------------------------------------------------

def _extract_compound_job_titles_regex(text: str) -> List[str]:
    """
    Extract common compound job titles such as 'operations coordinator'
    or 'private security'.
    """
    if not text:
        return []

    results = []
    seen = set()

    for regex in COMPOUND_JOB_TITLE_REGEXES:
        for match in regex.findall(text):
            title = " ".join(match.split()).strip()
            if not title:
                continue

            if not _is_valid_job_title(title):
                continue

            key = title.lower()
            if key not in seen:
                seen.add(key)
                results.append(title)

    return results

def _extract_simple_job_titles_from_lines(text: str) -> List[str]:
    if not text:
        return []

    results = []
    seen = set()

    for line in text.splitlines():
        line_clean = " ".join(line.split()).strip()
        if not line_clean:
            continue

        for rx, canonical_display in JOB_TITLE_REGEX_PATTERNS:
            if rx.search(line_clean):
                if not _is_valid_job_title(canonical_display):
                    continue

                key = canonical_display.lower()
                if key not in seen:
                    seen.add(key)
                    results.append(canonical_display)

    return results

def _ensure_default_degree(university: List[str], degree: List[str]) -> List[str]:
    """
    If university exists and no known degree level is found,
    add default Bachelor's Degree.
    """

    if not university:
        return degree

    if not degree:
        return ["Bachelor’s Degree"]

    degree_text = " ".join(degree).lower()

    if not any(level in degree_text for level in DEGREE_LEVELS):
        return degree + ["Bachelor’s Degree"]

    return degree

def _cleanup_degree_phrase(text: str) -> str:
    """
    Remove duplicated or noisy 'Degree' tokens after normalization.

    Examples:
        "Bachelor’s Degree Degree" -> "Bachelor’s Degree"
        "Master’s Degree’s Degree" -> "Master’s Degree"
        "Bachelor Degree" -> "Bachelor’s Degree"  (optional normalization)
    """
    if not text:
        return text

    t = text

    # Normalize common noisy variants (optional but helpful)
    # e.g. "Bachelor Degree" -> "Bachelor’s Degree"
    t = re.sub(r"(?i)\b(bachelor)\s+degree\b", "Bachelor’s Degree", t)
    t = re.sub(r"(?i)\b(master)\s+degree\b", "Master’s Degree", t)
    t = re.sub(r"(?i)\b(doctor)\s+degree\b", "Doctorate", t)

    # Remove repeated "Degree" tokens (case-insensitive)
    # e.g. "Degree Degree" / "Degree’s Degree" / "Degree's Degree"
    t = re.sub(r"(?i)\bdegree(?:['’]s)?\b(?:\s+\bdegree(?:['’]s)?\b)+", "Degree", t)

    # Fix the frequent bug: "Bachelor’s Degree Degree" -> "Bachelor’s Degree"
    # Keep the first and drop the duplicated trailing Degree
    t = re.sub(r"(?i)\b(bachelor(?:['’]s)?\s+degree)\s+degree\b", r"\1", t)
    t = re.sub(r"(?i)\b(master(?:['’]s)?\s+degree)\s+degree\b", r"\1", t)

    # Cleanup extra spaces
    t = re.sub(r"\s+", " ", t).strip()

    return t

def _normalize_degree_entries(degrees: List[str]) -> List[str]:
    """
    Normalize degree abbreviations/translations while preserving the remaining text.

    Examples:
        "BSc in Computer Science" -> "Bachelor of Science in Computer Science"
        "Licenciatura em Engenharia Informática" -> "Bachelor’s Degree em Engenharia Informática"
        "M.Sc Data Science" -> "Master of Science Data Science"
    """
    if not degrees:
        return []

    # Prefer longer keys first (e.g., "mestrado integrado" before "mestrado")
    keys_sorted = sorted(DEGREE_NORMALIZATION_MAP.keys(), key=len, reverse=True)

    normalized: List[str] = []

    for degree in degrees:
        original = degree.strip()
        if not original:
            continue

        text = original
        lower = text.lower()

        replaced_any = False

        for key in keys_sorted:
            full_form = DEGREE_NORMALIZATION_MAP[key]

            # Replace occurrences as whole tokens (case-insensitive)
            pattern = rf"(?i)\b{re.escape(key)}\b"

            if re.search(pattern, lower):
                text = re.sub(pattern, full_form, text)
                replaced_any = True
                lower = text.lower()  # keep in sync after replacement

        # Cleanup spacing around punctuation
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:])", r"\1", text)  # remove spaces before punctuation

        # Remove duplicated Degree tokens after normalization
        text = _cleanup_degree_phrase(text)

        normalized.append(text if (replaced_any or text) else original)

    # Deduplicate case-insensitively while preserving order
    seen = set()
    out: List[str] = []
    for item in normalized:
        k = item.lower()
        if k not in seen:
            seen.add(k)
            out.append(item)

    return out

def _is_valid_skill(text: str) -> bool:
    """
    Validate if a skill candidate is acceptable.

    Removes:
    - Languages
    - Generic placeholders
    - Academic degrees wrongly classified as skills
    - Language levels / certification levels
    - Generic proficiency phrases
    """

    if not text:
        return False

    t = text.lower().strip()

    # Remove academic degree phrases
    for keyword in ACADEMIC_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", t):
            return False

    # Remove languages
    if t in LANGUAGES_SET:
        return False

    for lang in LANGUAGES_SET:
        if re.search(rf"\b{re.escape(lang)}\b", t):
            return False

    GENERIC_BAD_SKILLS = {
        "academic background","academic","background","education","experience",
        "professional experience","hobbies","soft skills","hard skills","skills",
        "skill","fundamental","advanced user","basic user","independent user",
        "proficient user","user", "skill level", "level 1", "level 2", "level 3",
        "level 4", "level 5", "level 6", "level 7"
    }

    if t in GENERIC_BAD_SKILLS:
        return False

    # Language / certificate levels
    bad_skill_patterns = [
        r"^\s*[abc][12]\s*$",                  # C1, C2, B1, B2, A1, A2
        r"^\s*[abc][12]\s+and\s+[abc][12]\s*$",  # C1 and C2
        r"^\s*[abc][12]\s*/\s*[abc][12]\s*$",    # C1/C2
        r"\bfirst certificate\b",
        r"\badvanced user\b",
        r"\bbasic user\b",
        r"\bindependent user\b",
        r"\bproficient user\b",
        r"\bfundamental\b",
        r"^\s*level\s*\d+\s*$",     # Level 5
        r"^\s*skill\s+level\s*$",   # Skill Level
        r"\blevel\s*\d+\b",         # qualquer menção isolada de nível
        r"\bskill\s+level\b",       # expressão genérica
    ]

    if any(re.search(pattern, t, re.IGNORECASE) for pattern in bad_skill_patterns):
        return False

    if len(t) <= 2:
        return False

    return True

def _format_name_case(name: str) -> str:
    """
    Convert a name to title case token by token.
    Example:
        'GUILHERME RIBERT' -> 'Guilherme Ribert'
    """
    if not name:
        return name

    tokens = name.split()
    return " ".join(token.capitalize() for token in tokens)

def _clean_person_name(entities: List[str]) -> List[str]:
    """
    Clean and normalize person name entities.
    Prefer pairs of plausible person-name tokens and discard likely aliases/usernames.
    """
    cleaned_results = []

    suspicious_name_tokens = {
        "cv", "resume", "perfil", "profile", "user", "username",
        "linkedin", "github", "gitlab", "instagram", "facebook",
        "twitter", "x", "discord", "skype"
    }

    def looks_like_alias(token: str) -> bool:
        t = token.strip()

        if not t:
            return True

        lower = t.lower()

        if lower in suspicious_name_tokens:
            return True

        # very short weird tokens
        if len(t) <= 2:
            return True

        # tokens with no vowels are often handles/abbreviations
        if not re.search(r"[aeiouáéíóúàâêôãõäëïöü]", lower):
            return True

        # mixed odd casing like sNovaisg / mSd / xXfoo
        if not (t.isupper() or t.islower() or t.istitle()):
            return True

        return False

    for raw in entities:
        tokens = raw.replace(",", " ").split()
        tokens = [t for t in tokens if NAME_TOKEN_PATTERN.match(t)]

        if not tokens:
            continue

        # remove obvious aliases/usernames
        tokens = [t for t in tokens if not looks_like_alias(t)]

        if not tokens:
            continue

        # one-token case
        if len(tokens) == 1:
            if len(tokens[0]) >= 3:
                cleaned_results.append(_format_name_case(tokens[0]))
            continue

        # remove initials
        tokens = [t for t in tokens if len(t) > 1]

        if len(tokens) < 2:
            continue

        # normalize casing
        tokens = [t.capitalize() for t in tokens]

        # choose the first plausible adjacent pair
        chosen_pair = None
        for i in range(len(tokens) - 1):
            first = tokens[i]
            second = tokens[i + 1]

            if len(first) < 3 or len(second) < 3:
                continue

            chosen_pair = _format_name_case(f"{first} {second}")
            break

        if chosen_pair:
            cleaned_results.append(chosen_pair)

    return list(dict.fromkeys(cleaned_results))

def extract_top_name_candidate(text: str, window: int = 300) -> Optional[str]:
    """
    Strong fallback name extractor.
    Looks for a two-token Proper Case pattern inside the top section.
    """

    top = text[:window]

    # Normalize dots
    top = re.sub(r"\.", " ", top)

    lines = [l.strip() for l in top.split("\n") if l.strip()]

    forbidden_words = {
        "area", "interests", "objective", "profile", "summary",
        "university", "college", "school", "academy",
        "manager", "engineer", "project",
        "certificate", "department",
        "father", "mother", "nationality",
        "address", "passport",
        "seeking", "position", "experience",
        "science", "data", "starter", "starters", "ldsa",
        "cambridge", "assessment", "english", "fct", "unl"}

    noise_patterns = [
        r"@",
        r"\+?\d",
        r"\bmobile\b",
        r"\bemail\b",
        r"\bskype\b",
        r"\bphone\b",
    ]

    for line in lines[:8]:

        if any(re.search(p, line, re.IGNORECASE) for p in noise_patterns):
            continue

        tokens = line.split()

        # Only alphabetic tokens
        tokens = [t for t in tokens if NAME_TOKEN_PATTERN.match(t)]

        if len(tokens) < 2:
            continue
        

        # Slide window of 2 tokens inside the line
        for i in range(len(tokens) - 1):
            first = tokens[i]
            second = tokens[i + 1]
            
            if len(first) == 1 or len(second) == 1:
                continue

            # Both must start uppercase
            if not (first[0].isupper() and second[0].isupper()):
                continue

            if first.lower() in forbidden_words:
                continue

            if second.lower() in forbidden_words:
                continue

            return _format_name_case(f"{first} {second}")

    return None

def _remove_substring_duplicates(values: List[str]) -> List[str]:
    """
    Remove entities that are strict substrings of longer entities.
    Keeps the most specific version of overlapping entities.

    Example:
        ["FCT NOVA", "NOVA"] → ["FCT NOVA"]
    """

    values_sorted = sorted(values, key=len, reverse=True)
    filtered = []

    for val in values_sorted:
        if not any(val != other and val.lower() in other.lower() for other in filtered):
            filtered.append(val)

    return filtered

def _choose_first_address(addresses: List[str]) -> List[str]:
    """
    Choose the most complete address.
    Priority:
        1. Longest address
        2. First if equal length
    """
    if not addresses:
        return []

    valid = [a.strip() for a in addresses if isinstance(a, str) and a.strip()]
    if not valid:
        return []

    # Escolhe o mais longo
    best = max(valid, key=len)

    return [best]

def _expand_address_smart(full_text: str, address_fragment: str, window: int = 80, capitals_names: set|None = CAPITALS_NAMES, countries_names: set|None=COUNTRIES_NAMES) -> str:
    """
    Expand an address fragment by searching nearby for:
        - Postal codes
        - Street numbers
        - Capitals
        - Country names

    Keeps the closest structural signals.
    """
    COUNTRIES = set(COUNTRIES_AND_CAPITALS.keys())
    CAPITALS = {v for v in COUNTRIES_AND_CAPITALS.values() if v}
    match = re.search(re.escape(address_fragment), full_text, re.IGNORECASE)
    if not match:
        return address_fragment

    start, end = match.span()

    left_bound = max(0, start - window)
    right_bound = min(len(full_text), end + window)

    expanded_start = start
    expanded_end = end

    # Search right side
    right_text = full_text[end:right_bound]
    right_text_lower = right_text.lower()

    postal = POSTAL_PATTERN.search(right_text)
    if postal:
        expanded_end = end + postal.end()

    capitals = capitals_names if capitals_names is not None else CAPITALS
    countries = countries_names if countries_names is not None else COUNTRIES

    for name in capitals.union(countries):
        name_lower = name.lower()
        idx = right_text_lower.find(name_lower)
        if idx != -1:
            expanded_end = max(expanded_end, end + idx + len(name))

    # Search left side
    left_text = full_text[left_bound:start]

    number_matches = list(NUMBER_PATTERN.finditer(left_text))
    if number_matches:
        expanded_start = left_bound + number_matches[-1].start()

    expanded = full_text[expanded_start:expanded_end]

    expanded = re.sub(r"\s+", " ", expanded)
    expanded = expanded.strip(" ,;")

    return expanded

def _is_valid_job_title(text: str, threshold: float = 0.4) -> bool:
    """
    Returns False for generic section titles, language/certificate labels,
    and strings with too many non-letter characters.
    """
    if not text:
        return False

    t = text.strip().lower()

    generic_bad_titles = {
        "professional experience","experience","work experience","professional summary",
        "summary","profile","about me","education","academic background","skills",
        "technical skills","languages","certifications","projects","contacts","contact",
    }

    if t in generic_bad_titles:
        return False

    # reject titles that are too generic if they contain these patterns
    bad_title_patterns = [
        r"\bexperience\b",
        r"\bsummary\b",
        r"\bprofile\b",
        r"\beducation\b",
        r"\bskills\b",
        r"\blanguages\b",
        r"\bcertifications?\b",
    ]

    if any(re.search(pattern, t, re.IGNORECASE) for pattern in bad_title_patterns):
        return False

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False

    total_chars = len(chars)

    non_letter_chars = sum(
        1 for c in chars
        if not c.isalpha()
    )

    ratio = non_letter_chars / total_chars

    return ratio <= threshold

def _map_label_to_group(label: str) -> Optional[str]:
    """
    Converte a label bruta do GLiNER no grupo final usado no pipeline.
    """
    for group_name, group_labels in ENTITY_GROUPS.items():
        if label in group_labels:
            return group_name
    return None


def _normalizar_entidade_para_conflito(texto: str) -> str:
    """
    Normalização leve para comparar entidades iguais vindas de classes diferentes.
    """
    return " ".join(texto.strip().lower().split())


def _resolver_conflitos_por_score(
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Se o mesmo texto aparecer em 2+ classes, mantém só a previsão com maior score.

    Critério principal:
    - maior score

    Critério de desempate:
    - ordem fixa de prioridade
    """
    if not entities:
        return []

    # prioridade só para desempate
    priority = {
        "person_name": 7,
        "job_title": 6,
        "company": 5,
        "university": 4,
        "degree": 3,
        "address": 2,
        "skills": 1,
    }

    best_by_text: Dict[str, Dict[str, Any]] = {}

    for ent in entities:
        raw_text = str(ent.get("text", "")).strip()
        label = str(ent.get("label", "")).strip()
        score = float(ent.get("score", 0.0))

        if not raw_text or not label:
            continue

        group = _map_label_to_group(label)
        if group is None:
            continue

        key = _normalizar_entidade_para_conflito(raw_text)

        candidate = {
            **ent,
            "_group": group,
            "_score": score,
        }

        current = best_by_text.get(key)
        if current is None:
            best_by_text[key] = candidate
            continue

        current_score = float(current.get("_score", 0.0))
        current_group = str(current.get("_group", ""))

        if score > current_score:
            best_by_text[key] = candidate
        elif score == current_score:
            if priority.get(group, 0) > priority.get(current_group, 0):
                best_by_text[key] = candidate

    return list(best_by_text.values())


# ----------------------------------------------------------
# Filters for numbers and symbols
# ----------------------------------------------------------

def _is_allowed_numeric_symbol_exception(text: str) -> bool:
    """
    Exceções permitidas mesmo contendo muitos números/símbolos:
    - 12º ano / 12o ano / 12° ano
    - 2D / 3D
    """
    if not text:
        return False

    t = " ".join(text.strip().split())

    allowed_patterns = [
        # Escolaridade
        r"(?i)^\d{1,2}\s*[º°o]\s*ano$",          # 12º ano / 12o ano / 12° ano

        # Dimensão / formato
        r"(?i)^[23]\s*d$",                       # 2D / 3D

        # Linguagens / stacks curtas com símbolos
        r"(?i)^c\+$",                            # C+
        r"(?i)^c\+\+$",                          # C++
        r"(?i)^c#$",                             # C#
        r"(?i)^f#$",                             # F#
        r"(?i)^q#$",                             # Q#
        r"(?i)^a#?$",                            # A / A#
        r"(?i)^vb\.?net$",                       # VB.NET / VBNET
        r"(?i)^asp\.?net$",                      # ASP.NET / ASPNET
        r"(?i)^ado\.?net$",                      # ADO.NET / ADONET
        r"(?i)^net$",                            # NET
        r"(?i)^\.net$",                          # .NET

        # Web / frontend / backend
        r"(?i)^node\.?js$",                      # Node.js / NodeJS
        r"(?i)^next\.?js$",                      # Next.js / NextJS
        r"(?i)^nuxt\.?js$",                      # Nuxt.js / NuxtJS
        r"(?i)^vue\.?js$",                       # Vue.js / VueJS
        r"(?i)^react\.?js$",                     # React.js / ReactJS
        r"(?i)^angular\.?js$",                   # Angular.js / AngularJS
        r"(?i)^express\.?js$",                   # Express.js / ExpressJS
        r"(?i)^socket\.?io$",                    # Socket.IO / SocketIO

        # Outras tecnologias comuns
        r"(?i)^d3\.?js$",                        # D3.js / D3JS
        r"(?i)^chart\.?js$",                     # Chart.js / ChartJS
        r"(?i)^three\.?js$",                     # Three.js / ThreeJS
        r"(?i)^svelte\.?js$",                    # Svelte.js / SvelteJS
        r"(?i)^gatsby\.?js$",                    # Gatsby.js / GatsbyJS
        r"(?i)^tensorflow\.?js$",                # TensorFlow.js / TensorFlowJS

        # DevOps / dados / cloud
        r"(?i)^ms-?sql$",                        # MSSQL / MS-SQL
        r"(?i)^t-?sql$",                         # T-SQL
        r"(?i)^pl/?sql$",                        # PL/SQL
        r"(?i)^postgre\s*sql$",                  # Postgre SQL
        r"(?i)^postgre-?sql$",                   # Postgre-SQL
        r"(?i)^my\s*sql$",                       # MySQL escrito separado
        r"(?i)^no\s*sql$",                       # NoSQL escrito separado
    ]

    return any(re.fullmatch(pattern, t) for pattern in allowed_patterns)


def _has_more_than_60_percent_numbers_or_symbols(text: str) -> bool:
    """
    True se mais de 60% dos caracteres relevantes forem números ou símbolos.

    - ignora espaços
    - considera como "problemáticos":
        * dígitos
        * símbolos/pontuação
    - letras NÃO contam como problemáticas
    - exceções permitidas: 12º ano, 2D, 3D
    """
    if not text:
        return False

    if _is_allowed_numeric_symbol_exception(text):
        return False

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False

    problematic = sum(
        1 for c in chars
        if c.isdigit() or (not c.isalpha() and not c.isdigit())
    )

    return (problematic / len(chars)) > 0.6

# ----------------------------------------------------------
# Regex experience
# ----------------------------------------------------------
MONTH_PATTERN = (
    r"Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|"
    r"Sep|Sept|September|Oct|October|Nov|November|Dec|December|"
    r"Janeiro|Fev|Fevereiro|Março|Marco|Abr|Abril|Mai|Maio|Junho|Julho|Ago|Agosto|"
    r"Set|Setembro|Out|Outubro|Novembro|Dez|Dezembro|"
    r"Ene|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Septiembre|Octubre|Noviembre|Dic|Diciembre"
)

PRESENT_PATTERN = r"""Present|Current|Now|Today|Presente|Atual|Agora|Hoje|Actual|Ahora|Hoy"""

RANGE_CONNECTOR_PATTERN = r"""-|–|—|to|until|through|a|até|hasta"""

SINCE_PATTERN = r"""since|from|desde"""

MONTH_MAP = {
    # English
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,

    # Portuguese
    "fev": 2, "fevereiro": 2,
    "março": 3, "marco": 3,
    "abr": 4, "abril": 4,
    "mai": 5, "maio": 5,
    "junho": 6,
    "julho": 7,
    "ago": 8, "agosto": 8,
    "set": 9, "setembro": 9,
    "out": 10, "outubro": 10,
    "novembro": 11,
    "dez": 12, "dezembro": 12,

    # Spanish
    "ene": 1, "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "dic": 12, "diciembre": 12,
}

PRESENT_TOKENS = {
    # EN
    "present", "current", "now", "today", "nowadays", "contemporary",

    # PT
    "presente", "atual", "agora", "hoje",

    # ES
    "actual", "ahora", "hoy", "presente",

    # FR
    "présent", "present", "actuel", "aujourd’hui", "aujourd'hui", "maintenant",

    # DE
    "heute", "aktuell", "derzeit", "jetzt",

    # SV
    "nuvarande", "idag", "nu",

    # IT
    "attuale", "oggi", "adesso", "presente",

    # NL
    "heden", "vandaag", "nu",

    # PL
    "obecnie", "teraz", "dziś", "dzisiaj",

    # TR
    "günümüz", "gunumuz", "bugün", "bugun", "şimdi", "simdi", "mevcut",

    # RU
    "сейчас", "сегодня", "настоящее время", "по настоящее время",

    # HI
    "वर्तमान", "अभी", "आज",

    # ZH
    "至今", "现在", "目前", "今",

    # JA
    "現在", "今", "現在まで",

    # KO
    "현재", "지금", "오늘",
}

DATE_RANGE_REGEX = re.compile(
    r"""
    (?:
        # optional prefixes
        since|from|year\s+|
        desde|de|
        depuis|seit|sedan|dal|da|sinds|od|beri|itibaren|с|से|自|从|から|부터
    )?
    \s*

    # ---------------- START DATE ----------------
    (?:
        (?:\d{1,2}[./-]){0,2}
        (?P<start_month>
            Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|
            January|February|March|April|June|July|August|September|October|November|December|

            Janeiro|Fevereiro|Março|Marco|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro|
            Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez|

            Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|
            Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic|

            Janvier|Février|Fevrier|Mars|Avril|Mai|Juin|Juillet|Août|Aout|Septembre|Octobre|Novembre|Décembre|Decembre|
            Janv|Févr|Fevr|Avr|Juil|Déc|Dec|

            Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|
            Mär|Maerz|Okt|Dez|

            Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre|
            Gen|Mag|Giu|Lug|Set|Ott|

            Januari|Februari|Maart|April|Mei|Juni|Juli|Augustus|September|Oktober|November|December|
            Mrt|Mei|Okt|

            Styczeń|Styczen|Luty|Marzec|Kwiecień|Kwiecien|Maj|Czerwiec|Lipiec|Sierpień|Sierpien|Wrzesień|Wrzesien|Październik|Pazdziernik|Listopad|Grudzień|Grudzien|
            Sty|Lut|Kwi|Cze|Lip|Sie|Wrz|Paź|Paz|Lis|Gru|

            Ocak|Şubat|Subat|Mart|Nisan|Mayıs|Mayis|Haziran|Temmuz|Ağustos|Agustos|Eylül|Eylul|Ekim|Kasım|Kasim|Aralık|Aralik|
            Oca|Şub|Sub|Nis|Haz|Tem|Ağu|Agu|Eyl|Eki|Kas|Ara
        )?
        \s*
        (?P<start_year>\d{2,4})
        \s*(?:年|년)?
    )

    \s*
    (?:
        -|–|—|~|→|
        \bto\b|\buntil\b|\btill\b|\bthrough\b|
        \ba\b|\baté\b|\bhasta\b|\bà\b|\bau\b|\bbis\b|\btot\b|\bdo\b|\bkadar\b|\bпо\b|\bдо\b|\bतक\b|
        到|至|〜|～|까지
    )
    \s*

    # ---------------- END DATE ----------------
    (?:
        (?:\d{1,2}[./-]){0,2}
        (?P<end_month>
            Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|
            January|February|March|April|June|July|August|September|October|November|December|

            Janeiro|Fevereiro|Março|Marco|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro|
            Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez|

            Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|
            Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic|

            Janvier|Février|Fevrier|Mars|Avril|Mai|Juin|Juillet|Août|Aout|Septembre|Octobre|Novembre|Décembre|Decembre|
            Janv|Févr|Fevr|Avr|Juil|Déc|Dec|

            Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|
            Mär|Maerz|Okt|Dez|

            Gennaio|Febbraio|Marzo|Aprile|Maggio|Giugno|Luglio|Agosto|Settembre|Ottobre|Novembre|Dicembre|
            Gen|Mag|Giu|Lug|Set|Ott|

            Januari|Februari|Maart|April|Mei|Juni|Juli|Augustus|September|Oktober|November|December|
            Mrt|Mei|Okt|

            Styczeń|Styczen|Luty|Marzec|Kwiecień|Kwiecien|Maj|Czerwiec|Lipiec|Sierpień|Sierpien|Wrzesień|Wrzesien|Październik|Pazdziernik|Listopad|Grudzień|Grudzien|
            Sty|Lut|Kwi|Cze|Lip|Sie|Wrz|Paź|Paz|Lis|Gru|

            Ocak|Şubat|Subat|Mart|Nisan|Mayıs|Mayis|Haziran|Temmuz|Ağustos|Agustos|Eylül|Eylul|Ekim|Kasım|Kasim|Aralık|Aralik|
            Oca|Şub|Sub|Nis|Haz|Tem|Ağu|Agu|Eyl|Eki|Kas|Ara
        )?
        \s*
        (?P<end_year>
            \d{2,4}|
            Present|Current|Today|Now|
            Presente|Atual|Hoje|Agora|
            Actual|Hoy|Ahora|
            Présent|Actuel|Aujourd['’]hui|Maintenant|
            Heute|Aktuell|Derzeit|Jetzt|
            Nuvarande|Idag|Nu|
            Attuale|Oggi|Adesso|
            Heden|Vandaag|
            Obecnie|Teraz|Dziś|Dzisiaj|
            Günümüz|Gunumuz|Bugün|Bugun|Şimdi|Simdi|Mevcut|
            Настоящее\s+время|Сейчас|Сегодня|По\s+настоящее\s+время|
            वर्तमान|अभी|आज|
            至今|现在|目前|今|
            現在|現在まで|
            현재|지금|오늘
        )
        \s*(?:年|년)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

SINCE_REGEX = re.compile(
    r"""
    (?:
        since|from|
        desde|
        depuis|
        seit|
        sedan|
        dal|da|
        sinds|
        od|
        beri|itibaren|
        с|
        से|
        自|从|
        から|
        부터
    )
    \s*
    (?P<start_month>
        Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|
        January|February|March|April|June|July|August|September|October|November|December|

        Janeiro|Fevereiro|Março|Marco|Abril|Maio|Junho|Julho|Agosto|Setembro|Outubro|Novembro|Dezembro|
        Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez|

        Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Octubre|Noviembre|Diciembre|
        Ene|Feb|Mar|Abr|May|Jun|Jul|Ago|Sep|Oct|Nov|Dic
    )?
    \.?
    \s*
    (?P<start_year>\d{2,4})
    \s*(?:年|년)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

EXPLICIT_YEARS_REGEX = re.compile(
    r"""
    (?:
        # EN
        less\ than|under|more\ than|over|at\ least|minimum|min\.?|up\ to|approx\.?|around|about|
        # PT
        menos\ de|mais\ de|pelo\ menos|cerca\ de|aprox\.?|
        # ES
        menos\ de|más\ de|al\ menos|aprox\.?|cerca\ de|
        # FR
        moins\ de|plus\ de|au\ moins|environ|
        # DE
        weniger\ als|mehr\ als|mindestens|ca\.?|
        # SV
        mindre\ än|mer\ än|minst|cirka|
        # IT
        meno\ di|più\ di|almeno|circa|
        # NL
        minder\ dan|meer\ dan|minstens|ongeveer|
        # PL
        mniej\ niż|więcej\ niż|co\ najmniej|około|
        # TR
        en\ az|yaklaşık
    )?
    \s*

    (?P<years>\d{1,3})
    \+?
    \s*

    (?:
        # EN
        years?|yrs?|y|
        # PT
        anos?|ano|
        # ES
        años?|ano|
        # FR
        ans?|année?s?|
        # DE
        jahre?n?|jahr|
        # SV
        år|
        # IT
        anni|anno|
        # NL
        jaren?|jaar|
        # PL
        lat|rok|lata|
        # TR
        yıl|yil|
        # RU
        лет|года?|год|
        # HI
        साल|
        # ZH
        年|
        # JA
        年|
        # KO
        년
    )

    (?:
        \s*
        (?:
            # EN
            of\s+experience|experience|exp\.?|
            # PT
            de\s+experiência|experiência|exp\.?|
            # ES
            de\s+experiencia|experiencia|exp\.?|
            # FR
            d['’]expérience|expérience|exp\.?|
            # DE
            erfahrung|berufserfahrung|
            # SV
            erfarenhet|
            # IT
            esperienza|
            # NL
            ervaring|
            # PL
            doświadczenia|
            # TR
            deneyim|
            # RU
            опыта|опыт|
            # HI
            अनुभव|
            # ZH
            经验|
            # JA
            経験|
            # KO
            경험
        )
    )?
    """,
    re.IGNORECASE | re.VERBOSE,
)

WORK_KEYWORDS = {
    "experience", "work", "employment", "career", "history",
    "job", "role", "company", "professional", "corporate",
    "intern", "internship",
    "engineer", "developer", "manager", "analyst",
    "consultant", "coordinator", "supervisor",
    "security", "guard", "officer",
    "driver", "deliveryman", "delivery",
    "lifeguard", "operations", "operator", "pilot",
    "construction", "site", "preparation",
    "technician", "assistant",
    "professor", "scientist", "teacher",
    "crew", "staff", "client", "clients",
    "speaker", "logistics", "deployment",
    "managed", "developed", "supervised", "led", "teaching",
}

EXPERIENCE_EDU_KEYWORDS = {
    "education", "academic", "student", "degree", "course",
    "bachelor", "master", "phd", "msc", "mba",
    "university", "academy", "school", "faculty",
    "graduated", "graduation", "thesis", "bootcamp",
}

NON_WORK_EXPERIENCE_KEYWORDS = {
    "project", "projects", "capstone", "training", "learning", "bootcamp",
    "skills", "tools", "technologies", "frameworks", "packages", "modeling",
    "volunteer", "sports", "hobbies","sailing",
    "music", "band", "bands", "runner", "running", "karate", "piano",
    "paraglider", "dancer", "scout",
    "non-corporate", "activities", "other activities",
}

WORK_SECTION_HEADERS = {
    "work experience",
    "professional experience",
    "corporate experience",
    "experiência profissional",
    "employment history",
    "career history",
    "professional background",
    "professional history",
    "professional career"
}

NON_WORK_SECTION_HEADERS = {
    "education",
    "education and training",
    "academic background",
    "projects",
    "project",
    "certifications",
    "skills",
    "competencies",
    "language skills",
    "languages",
    "volunteering",
    "other activities",
    "activities",
    "hobbies",
    "extra",
    "extra activities",
    "non-corporate experience",
    "music",
    "karaté",
    "karate",
    "runner",
}

# ----------------------------------------------------------
# Helpers to transform months and years
# ----------------------------------------------------------
def _parse_month(token: Optional[str], default: int) -> int:
    """
    Convert a month token to its numeric value.

    Examples:
        "Jan" -> 1
        "Maio" -> 5
        None -> default
        "unknown" -> default
    """
    if not token:
        return default

    normalized = token.strip().lower()
    return MONTH_MAP.get(normalized, default)


def _parse_year(token: Optional[str]) -> Optional[int]:
    """
    Convert a year token or present-token into a valid 4-digit year.
    Supports 2-digit and 4-digit years.
    """
    if not token:
        return None

    normalized = token.strip().lower()

    if normalized in PRESENT_TOKENS:
        return CURRENT_DATE.year

    if not re.fullmatch(r"\d{2,4}", normalized):
        return None

    year = int(normalized)

    if year < 100:
        year += 2000

    if 1950 <= year <= CURRENT_DATE.year + 1:
        return year

    return None

def _to_month_index(year: int, month: int) -> int:
    """
    Convert (year, month) to a sortable absolute month index.
    """
    return year * 12 + (month - 1)


def _merge_month_intervals(intervals: List[Tuple[int, int]]) -> int:
    """
    Merge overlapping month intervals and return total duration in months.

    Each interval is treated as [start, end), meaning end is exclusive.
    """
    if not intervals:
        return 0

    sorted_intervals = sorted(intervals)
    merged: List[Tuple[int, int]] = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return sum(max(0, end - start) for start, end in merged)

# ----------------------------------------------------------
# Helpers for Experience
# ----------------------------------------------------------
def _get_nearby_lines(text: str, pos: int, lines_before: int = 3, lines_after: int = 1) -> str:
    """
    Return nearby lines around a character position.
    Useful to detect section headers near a date range.
    """
    if not text:
        return ""

    lines = text.splitlines()

    char_count = 0
    line_idx = 0

    for i, line in enumerate(lines):
        char_count += len(line) + 1
        if char_count >= pos:
            line_idx = i
            break

    start = max(0, line_idx - lines_before)
    end = min(len(lines), line_idx + lines_after + 1)

    return "\n".join(lines[start:end]).lower()

def _classify_section_near_date(text: str, pos: int) -> str:
    """
    Classify a date by the nearest relevant section header above it.
    Returns:
    - 'work'
    - 'non_work'
    - ''
    """
    if not text:
        return ""

    lines = text.splitlines()

    char_count = 0
    line_idx = 0
    for i, line in enumerate(lines):
        char_count += len(line) + 1
        if char_count >= pos:
            line_idx = i
            break

    # look upwards for the nearest meaningful header
    for i in range(line_idx, -1, -1):
        line = " ".join(lines[i].split()).strip().lower()
        if not line:
            continue

        for header in NON_WORK_SECTION_HEADERS:
            if re.fullmatch(rf"{re.escape(header)}", line):
                return "non_work"

        for header in WORK_SECTION_HEADERS:
            if re.fullmatch(rf"{re.escape(header)}", line):
                return "work"

    return ""

def clean_text_keep_lines(text: str) -> str:
    """
    Normalize text while preserving line structure.

    - keeps line breaks
    - removes common bullet/tab noise
    - collapses repeated spaces
    - collapses excessive blank lines
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\uf0b7\u2022*]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    """
    Return the start and end indexes of the line containing `pos`.
    """
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)

    if line_end == -1:
        line_end = len(text)

    return line_start, line_end

def _match_has_experience_word(match_text: str) -> bool:
    if not match_text:
        return False

    return bool(re.search(
        r"""
        \b(
            experience|experiences|exp\.?|
            experiência|experiencias?|experiencia|
            expérience|
            erfahrung|berufserfahrung|
            erfarenhet|
            esperienza|
            ervaring|
            doświadczenia|
            deneyim|
            опыт|опыта|
            अनुभव|
            经验|経験|경험
        )\b
        """,
        match_text,
        re.IGNORECASE | re.VERBOSE,
    ))

def _get_local_context(
    text: str,
    pos: int,
    left_words: int = 25,
    right_words: int = 25,
) -> str:
    """
    Extract a word-based local context window around a character position.
    """
    words = re.split(r"\s+", text)

    char_count = 0
    idx = len(words) - 1 if words else 0

    for i, word in enumerate(words):
        char_count += len(word) + 1
        if char_count >= pos:
            idx = i
            break

    start = max(0, idx - left_words)
    end = min(len(words), idx + right_words)

    return " ".join(words[start:end]).lower()

def _tokenize_context(text: str) -> Set[str]:
    """
    Convert context text into a normalized token set.
    """
    return {
        re.sub(r"[^\w]", "", token.lower())
        for token in re.split(r"\s+", text)
        if token and re.sub(r"[^\w]", "", token.lower())
    }

def _is_valid_explicit_experience_match(
    text: str,
    pos: int,
    years: int,
    academic_keywords: Set[str],
    non_work_keywords: Set[str],
) -> bool:
    """
    Decide if an explicit expression like '5 years experience'
    is likely to refer to professional experience.
    """
    if years <= 0 or years > 50:
        return False

    local_ctx = _get_local_context(text, pos, left_words=30, right_words=30)

    hard_negative_patterns = [
        r"\bthesis experience\b",
        r"\bproject experience\b",
        r"\bprojects experience\b",
        r"\bnon-corporate\b",
        r"\bother activities\b",
        r"\bflying hours\b",
        r"\bkm per week\b",
        r"\bvolunteer\b",
        r"\bstudent\b",
        r"\bcourse\b",
        r"\bacademy\b",
        r"\bbootcamp\b",
    ]

    if any(re.search(pattern, local_ctx, re.IGNORECASE) for pattern in hard_negative_patterns):
        return False

    token_set = _tokenize_context(local_ctx)

    if token_set & academic_keywords:
        return False

    if token_set & non_work_keywords:
        return False

    work_signal_patterns = [
        r"\bwork(?:ing)? experience\b",
        r"\bprofessional experience\b",
        r"\bindustry experience\b",
        r"\bcorporate experience\b",
        r"\bcurrently working\b",
        r"\bjob\b",
        r"\brole\b",
        r"\bconsultant\b",
        r"\bengineer\b",
        r"\bscientist\b",
        r"\bdeveloper\b",
        r"\bmanager\b",
        r"\banalyst\b",
        r"\bprofessor\b",
        r"\bteacher\b",
        r"\bintern(?:ship)?\b",
        r"\bsupervisor\b",
        r"\bcoordinator\b",
        r"\bdriver\b",
        r"\bsecurity\b",
        r"\bguard\b",
        r"\bofficer\b",
        r"\blifeguard\b",
        r"\boperations?\b",
        r"\bconstruction\b",
        r"\bsite\b",
        r"\btechnician\b",
        r"\bassistant\b",
        r"\bpilot\b",
        r"\boperator\b",
        r"\bdeliveryman\b",
        r"\bdelivery\b",
    ]

    return any(re.search(pattern, local_ctx, re.IGNORECASE) for pattern in work_signal_patterns)

def _build_academic_context_keywords(
    base_education_keywords: Set[str],
    degrees: List[str] | None = None,
    universities: List[str] | None = None,
) -> Set[str]:
    keywords = set(base_education_keywords)

    for item in (degrees or []):
        for token in re.findall(r"\w+", item.lower(), flags=re.UNICODE):
            if len(token) >= 4:
                keywords.add(token)

    for item in (universities or []):
        for token in re.findall(r"\w+", item.lower(), flags=re.UNICODE):
            if len(token) >= 4:
                keywords.add(token)

    for key in DEGREE_NORMALIZATION_MAP.keys():
        for token in re.findall(r"\w+", key.lower(), flags=re.UNICODE):
            if len(token) >= 4:
                keywords.add(token)

    for level in DEGREE_LEVELS:
        keywords.add(level.lower())

    return keywords

def _context_has_keywords(
    text: str,
    pos: int,
    direction: str,
    window_words: int,
    academic_keywords: Set[str],
    work_keywords: Set[str],
) -> str:
    words = re.split(r"\s+", text)

    char_count = 0
    idx = len(words) - 1 if words else 0

    for i, word in enumerate(words):
        char_count += len(word) + 1
        if char_count >= pos:
            idx = i
            break

    if direction == "right":
        segment = words[idx: idx + window_words]
    elif direction == "left":
        segment = words[max(0, idx - window_words): idx]
    else:
        return ""

    segment_set = {
        re.sub(r"[^\w]", "", word.lower())
        for word in segment
        if word and re.sub(r"[^\w]", "", word.lower())
    }

    if segment_set & academic_keywords:
        return "edu"

    if segment_set & work_keywords:
        return "work"

    return ""

def _build_job_title_keywords(job_titles: List[str] | None = None) -> Set[str]:
    keywords: Set[str] = set()

    for item in (job_titles or []):
        for token in re.findall(r"\w+", item.lower(), flags=re.UNICODE):
            if len(token) >= 3:
                keywords.add(token)

    return keywords

def _nearby_lines_have_job_title(
    text: str,
    pos: int,
    job_titles: List[str] | None = None,
    window_lines: int = 2
) -> bool:
    """
    Check whether nearby lines around a date contain a likely job title.
    """
    if not text:
        return False

    lines = text.splitlines()

    char_count = 0
    line_idx = 0

    for i, line in enumerate(lines):
        char_count += len(line) + 1
        if char_count >= pos:
            line_idx = i
            break

    start = max(0, line_idx - window_lines)
    end = min(len(lines), line_idx + window_lines + 1)

    nearby_lines = []
    for line in lines[start:end]:
        line_clean = " ".join(line.split()).strip()
        if line_clean:
            nearby_lines.append(line_clean)

    # 1) direct match against extracted job titles
    extracted_titles = [jt.strip().lower() for jt in (job_titles or []) if jt and jt.strip()]
    for line_clean in nearby_lines:
        line_lower = line_clean.lower()

        if line_lower in extracted_titles:
            return True

        if any(jt in line_lower or line_lower in jt for jt in extracted_titles):
            return True

    # 2) fallback com o novo matching flexível
    nearby_text = "\n".join(nearby_lines)

    if _extract_simple_job_titles_from_lines(nearby_text):
        return True

    for line_clean in nearby_lines:
        if not _is_valid_job_title(line_clean):
            continue

        for rx in COMPOUND_JOB_TITLE_REGEXES:
            if rx.search(line_clean):
                return True

    return False

def dividir_texto_em_chunks(
    texto: str,
    max_chars: int = 650,
    overlap: int = 100,
) -> List[str]:
    """
    Divide texto longo em chunks com overlap.
    """
    texto = texto.strip()
    if not texto:
        return []

    blocos = re.split(r"\n+", texto)
    blocos = [b.strip() for b in blocos if b.strip()]

    chunks: List[str] = []
    chunk_atual = ""

    for bloco in blocos:
        candidato = f"{chunk_atual}\n{bloco}".strip() if chunk_atual else bloco

        if len(candidato) <= max_chars:
            chunk_atual = candidato
        else:
            if chunk_atual:
                chunks.append(chunk_atual)

            if len(bloco) <= max_chars:
                chunk_atual = bloco
            else:
                inicio = 0
                passo = max(1, max_chars - overlap)

                while inicio < len(bloco):
                    fim = inicio + max_chars
                    pedaco = bloco[inicio:fim].strip()
                    if pedaco:
                        chunks.append(pedaco)
                    inicio += passo

                chunk_atual = ""

    if chunk_atual:
        chunks.append(chunk_atual)

    return chunks

def _agrupar_entidades_dicts(entidades_chunks: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """
    Junta entidades vindas de vários chunks, removendo duplicados
    e preservando a ordem de primeira aparição.
    """
    labels = [
        "person_name",
        "job_title",
        "company",
        "university",
        "degree",
        "address",
        "skills",
    ]

    resultado: Dict[str, List[str]] = {label: [] for label in labels}
    vistos: Dict[str, Set[str]] = {label: set() for label in labels}

    for entidades in entidades_chunks:
        for label in labels:
            for valor in entidades.get(label, []):
                if not isinstance(valor, str):
                    continue

                v = " ".join(valor.split()).strip()
                if not v:
                    continue

                key = v.lower()
                if key not in vistos[label]:
                    vistos[label].add(key)
                    resultado[label].append(v)

    return resultado

def _apply_regex_fallbacks_to_entities(
    text: str,
    entities_by_label: Dict[str, List[str]],
    use_job_title_regex_fallback: bool = USE_JOB_TITLE_REGEX_FALLBACK,
    use_skill_regex_fallback: bool = USE_SKILL_REGEX_FALLBACK,
    use_software_regex_fallback: bool = USE_SOFTWARE_REGEX_FALLBACK,
    use_programming_regex_fallback: bool = USE_PROGRAMMING_REGEX_FALLBACK,
    use_general_skills_regex_fallback: bool = USE_GENERAL_SKILLS_REGEX_FALLBACK,
) -> Dict[str, List[str]]:
    if not text:
        return entities_by_label

    if use_job_title_regex_fallback:
        existing_job_titles = {t.lower() for t in entities_by_label["job_title"]}

        regex_job_titles: List[str] = []
        regex_job_titles.extend(_extract_simple_job_titles_from_lines(text))
        regex_job_titles.extend(_extract_compound_job_titles_regex(text))

        for title in regex_job_titles:
            key = title.lower()
            if key not in existing_job_titles:
                entities_by_label["job_title"].append(title)
                existing_job_titles.add(key)

    if use_skill_regex_fallback:
        existing_skills = {s.lower() for s in entities_by_label["skills"]}

        if use_software_regex_fallback:
            for software in _extract_software_as_skill_from_text(text):
                key = software.lower()
                if key not in existing_skills:
                    entities_by_label["skills"].append(software)
                    existing_skills.add(key)

        if use_programming_regex_fallback:
            for lib in _extract_programming_libraries_from_text(text):
                key = lib.lower()
                if key not in existing_skills:
                    entities_by_label["skills"].append(lib)
                    existing_skills.add(key)

        if use_general_skills_regex_fallback:
            for skill in _extract_general_skills_from_text(text):
                key = skill.lower()
                if key not in existing_skills:
                    entities_by_label["skills"].append(skill)
                    existing_skills.add(key)

    return entities_by_label

def _extrair_entidades_texto_longo(
    text: str,
    model: Optional[GLiNER] = None,
    threshold: float = 0.15,
    debug_save_json: bool = False,
    debug_output_path: Optional[Path | str] = OUTPUTS_DIR,
    use_regex_fallbacks: bool = USE_REGEX_FALLBACKS,
    use_job_title_regex_fallback: bool = USE_JOB_TITLE_REGEX_FALLBACK,
    use_skill_regex_fallback: bool = USE_SKILL_REGEX_FALLBACK,
    use_software_regex_fallback: bool = USE_SOFTWARE_REGEX_FALLBACK,
    use_programming_regex_fallback: bool = USE_PROGRAMMING_REGEX_FALLBACK,
    use_general_skills_regex_fallback: bool = USE_GENERAL_SKILLS_REGEX_FALLBACK,
    use_regex_fallbacks_on_chunks: bool = USE_REGEX_FALLBACKS_ON_CHUNKS,
) -> Dict[str, List[str]]:
    """
    Processa texto longo por chunks e agrega as entidades detetadas.

    Regra especial:
    - person_name é extraído apenas do topo do CV
    - restantes entidades são agregadas por chunks
    """
    empty_result = {
        "person_name": [],
        "job_title": [],
        "company": [],
        "university": [],
        "degree": [],
        "address": [],
        "skills": [],
    }

    if not text or not text.strip():
        return empty_result

    if model is None:
        model = get_gliner_model()

    # -------------------------------
    # 1) Nome: só no topo do CV
    # -------------------------------
    best_person_name = _extract_best_person_name_from_top(
        text=text,
        model=model,
        threshold=threshold,
        top_window=450,
    )

    chunks = dividir_texto_em_chunks(text)

    # Se houver apenas 1 chunk
    if len(chunks) <= 1:
        entidades = extract_entities_from_text(
            text,
            model=model,
            threshold=threshold,
            debug_save_json=debug_save_json,
            debug_output_path=debug_output_path,
            use_regex_fallbacks=use_regex_fallbacks,
            use_job_title_regex_fallback=use_job_title_regex_fallback,
            use_skill_regex_fallback=use_skill_regex_fallback,
            use_software_regex_fallback=use_software_regex_fallback,
            use_programming_regex_fallback=use_programming_regex_fallback,
            use_general_skills_regex_fallback=use_general_skills_regex_fallback,
        )
        entidades["person_name"] = best_person_name
        return entidades

    # -------------------------------
    # 2) Restantes entidades por chunk
    # -------------------------------
    entidades_chunks: List[Dict[str, List[str]]] = []

    for chunk in chunks:
        entidades = extract_entities_from_text(
            chunk,
            model=model,
            threshold=threshold,
            debug_save_json=debug_save_json,
            debug_output_path=debug_output_path,
            use_regex_fallbacks=use_regex_fallbacks if use_regex_fallbacks_on_chunks else False,
            use_job_title_regex_fallback=use_job_title_regex_fallback if use_regex_fallbacks_on_chunks else False,
            use_skill_regex_fallback=use_skill_regex_fallback if use_regex_fallbacks_on_chunks else False,
            use_software_regex_fallback=use_software_regex_fallback if use_regex_fallbacks_on_chunks else False,
            use_programming_regex_fallback=use_programming_regex_fallback if use_regex_fallbacks_on_chunks else False,
            use_general_skills_regex_fallback=use_general_skills_regex_fallback if use_regex_fallbacks_on_chunks else False,
        )

        entidades["person_name"] = []
        entidades_chunks.append(entidades)

    resultado = _agrupar_entidades_dicts(entidades_chunks)
    resultado["person_name"] = best_person_name

    if use_regex_fallbacks and not use_regex_fallbacks_on_chunks:
        resultado = _apply_regex_fallbacks_to_entities(
            text=text,
            entities_by_label=resultado,
            use_job_title_regex_fallback=use_job_title_regex_fallback,
            use_skill_regex_fallback=use_skill_regex_fallback,
            use_software_regex_fallback=use_software_regex_fallback,
            use_programming_regex_fallback=use_programming_regex_fallback,
            use_general_skills_regex_fallback=use_general_skills_regex_fallback,
        )

    return resultado


# ----------------------------------------------------------
# Experience functions
# ----------------------------------------------------------
def extract_experience_years(
    raw_text: str,
    degrees: List[str] | None = None,
    universities: List[str] | None = None,
    job_titles: List[str] | None = None,
) -> Optional[int]:
    """
    Estimate the total number of professional experience years from CV text.

    The estimation combines multiple complementary strategies:

    1) Explicit mentions of experience duration
       Examples:
           "5 years", "7+ years of experience", "over 3 yrs"

    2) Closed date ranges describing employment periods
       Examples:
           "2019–2024", "May 2017 - April 2019", "2020 to Present"

    3) Open date ranges using "since"
       Examples:
           "since 2020", "working since 2018"

    4) Semantic context filtering
       - Excludes education-related periods
       - Keeps work/professional periods
       - Uses:
           • short right-side context
           • long left-side context

    5) Interval merging
       - Prevents double counting overlapping jobs

    Returns
    -------
    Optional[int]
        Total estimated years of professional experience.
        Returns None if no reliable estimation is possible.
    """

    if not raw_text:
        return None

    text = clean_text_keep_lines(raw_text)

    academic_keywords = _build_academic_context_keywords(
        base_education_keywords=EXPERIENCE_EDU_KEYWORDS,
        degrees=degrees,
        universities=universities,
    )

    job_title_keywords = _build_job_title_keywords(job_titles)
    dynamic_work_keywords = set(WORK_KEYWORDS) | job_title_keywords

    # ------------------------------------------------------
    # 1) Explicit mentions (e.g. "5 years of experience")
    # ------------------------------------------------------
    explicit_years: List[int] = []
    line_best_matches: dict[tuple[int, int], re.Match[str]] = {}

    for match in EXPLICIT_YEARS_REGEX.finditer(text):
        pos = match.start()
        line_span = _line_bounds(text, pos)

        current_best = line_best_matches.get(line_span)
        if current_best is None:
            line_best_matches[line_span] = match
            continue

        current_has_exp = _match_has_experience_word(current_best.group(0))
        new_has_exp = _match_has_experience_word(match.group(0))

        # Prefer the one that explicitly includes "experience"
        if new_has_exp and not current_has_exp:
            line_best_matches[line_span] = match

    for match in line_best_matches.values():
        years = int(match.group("years"))
        pos = match.start()

        if _is_valid_explicit_experience_match(
            text=text,
            pos=pos,
            years=years,
            academic_keywords=academic_keywords,
            non_work_keywords=NON_WORK_EXPERIENCE_KEYWORDS,
        ):
            explicit_years.append(years)

    # ------------------------------------------------------
    # 2) Closed date ranges (start -> end)
    # ------------------------------------------------------
    intervals: List[Tuple[int, int]] = []

    for match in DATE_RANGE_REGEX.finditer(text):
        start_year = _parse_year(match.group("start_year"))
        end_year_raw = match.group("end_year")
        if not end_year_raw:
            continue

        end_year = _parse_year(end_year_raw)

        if start_year is None or end_year is None:
            continue

        start_month = _parse_month(match.group("start_month"), 1)

        is_present = end_year_raw.strip().lower() in PRESENT_TOKENS
        end_month = _parse_month(
            None if is_present else match.group("end_month"),
            CURRENT_DATE.month if is_present else 12,
        )

        start_idx = _to_month_index(start_year, start_month)
        end_idx = _to_month_index(end_year, end_month)

        # Invalid reversed range
        if end_idx < start_idx:
            continue

        pos = match.start()

        section_cls = _classify_section_near_date(text, pos)

        if section_cls == "non_work":
            continue

        if _nearby_lines_have_job_title(text, pos, job_titles=job_titles, window_lines=2):
            intervals.append((start_idx, end_idx + 1))
            continue

        local_ctx = _get_local_context(text, pos, left_words=40, right_words=20)
        token_set = _tokenize_context(local_ctx)

        if section_cls == "work":
            intervals.append((start_idx, end_idx + 1))
            continue

        if token_set & NON_WORK_EXPERIENCE_KEYWORDS:
            continue

        right_ctx = _context_has_keywords(
            text=text,
            pos=pos,
            direction="right",
            window_words=12,
            academic_keywords=academic_keywords,
            work_keywords=dynamic_work_keywords,
        )

        if right_ctx == "edu":
            continue

        if right_ctx == "work":
            intervals.append((start_idx, end_idx + 1))
            continue

        left_ctx = _context_has_keywords(
            text=text,
            pos=pos,
            direction="left",
            window_words=100,
            academic_keywords=academic_keywords,
            work_keywords=dynamic_work_keywords,
        )

        if left_ctx == "work":
            intervals.append((start_idx, end_idx + 1))

    # ------------------------------------------------------
    # 3) Open intervals like "since 2020"
    # ------------------------------------------------------
    for match in SINCE_REGEX.finditer(text):
        start_year = _parse_year(match.group("start_year"))
        if start_year is None:
            continue

        start_month = _parse_month(match.group("start_month"), 1)
        start_idx = _to_month_index(start_year, start_month)
        end_idx = _to_month_index(CURRENT_DATE.year, CURRENT_DATE.month)

        if end_idx < start_idx:
            continue

        pos = match.start()

        section_cls = _classify_section_near_date(text, pos)

        if section_cls == "non_work":
            continue

        if _nearby_lines_have_job_title(text, pos, job_titles=job_titles, window_lines=2):
            interval = (start_idx, end_idx + 1)
            if interval not in intervals:
                intervals.append(interval)
            continue

        local_ctx = _get_local_context(text, pos, left_words=40, right_words=20)
        token_set = _tokenize_context(local_ctx)

        if section_cls == "work":
            interval = (start_idx, end_idx + 1)
            if interval not in intervals:
                intervals.append(interval)
            continue

        if token_set & NON_WORK_EXPERIENCE_KEYWORDS:
            continue

        right_ctx = _context_has_keywords(
            text=text,
            pos=pos,
            direction="right",
            window_words=12,
            academic_keywords=academic_keywords,
            work_keywords=dynamic_work_keywords,
        )

        if right_ctx == "edu":
            continue

        left_ctx = _context_has_keywords(
            text=text,
            pos=pos,
            direction="left",
            window_words=100,
            academic_keywords=academic_keywords,
            work_keywords=dynamic_work_keywords,
        )

        if right_ctx == "work" or left_ctx == "work":
            interval = (start_idx, end_idx + 1)
            if interval not in intervals:
                intervals.append(interval)

    # ------------------------------------------------------
    # 4) Merge intervals and compute total months
    # ------------------------------------------------------
    total_months = _merge_month_intervals(intervals)

    if total_months < 0:
        return None

    # Reject unrealistic values
    if total_months > 50 * 12:
        return None

    # ------------------------------------------------------
    # 5) Fallback to explicit mentions if no interval found
    # ------------------------------------------------------
    if total_months == 0:
        if explicit_years:
            best_explicit = max(explicit_years)
            if best_explicit <= 15:
                return best_explicit
        return None

    # Convert merged months to years
    return int(total_months / 12 + 0.5)


# ----------------------------------------------------------
# MAIN FUNCTIONS
# ----------------------------------------------------------
def extract_entities_from_text(
    text: str,
    model: Optional[GLiNER] = None,
    threshold: float = 0.15,
    debug_save_json: bool = False,
    debug_output_path: Optional[Path | str] = OUTPUTS_DIR,
    use_regex_fallbacks: bool = USE_REGEX_FALLBACKS,
    use_job_title_regex_fallback: bool = USE_JOB_TITLE_REGEX_FALLBACK,
    use_skill_regex_fallback: bool = USE_SKILL_REGEX_FALLBACK,
    use_software_regex_fallback: bool = USE_SOFTWARE_REGEX_FALLBACK,
    use_programming_regex_fallback: bool = USE_PROGRAMMING_REGEX_FALLBACK,
    use_general_skills_regex_fallback: bool = USE_GENERAL_SKILLS_REGEX_FALLBACK,
) -> Dict[str, List[str]]:

    result_labels = [
                "person_name",
                "job_title",
                "company",
                "university",
                "degree",
                "address",
                "skills"]

    if not text or not text.strip():
        return {label: [] for label in result_labels}

    if model is None:
        model = get_gliner_model()

    model = cast(GLiNER, model)

    try:
        # ---------------------------------------
        # Get broader predictions with a lower threshold, then filter manually
        # ---------------------------------------
        raw_threshold = 0.10

        all_entities = model.predict_entities(
            text,
            ALL_LABELS,
            threshold=raw_threshold
        )  # type: ignore[attr-defined]
        # ---------------------------------------
        # Optional debug JSON save
        # ---------------------------------------
        if debug_save_json:
            import json

            debug_data = []

            for ent in all_entities:
                debug_data.append({
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": float(ent.get("score", 0.0))
                })

            if debug_output_path:
                output_path = Path(debug_output_path)

                if output_path.is_dir() or not output_path.suffix:
                    output_file = output_path / "gliner_thresholds.json"
                else:
                    output_file = output_path
            else:
                output_file = Path(OUTPUTS_DIR) / "gliner_thresholds.json"

            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, indent=4, ensure_ascii=False)

        # ---------------------------------------
        # Apply threshold manually
        # ---------------------------------------
        entities_filtradas = []

        for ent in all_entities:
            label = ent.get("label")
            score = float(ent.get("score", 0.0))

            # person name: threshold mais permissivo
            if label in ENTITY_GROUPS["person_name"]:
                if score >= 0.1:
                    entities_filtradas.append(ent)
            else:
                if score >= threshold:
                    entities_filtradas.append(ent)

        # Resolve conflitos entre classes para o mesmo texto
        entities = _resolver_conflitos_por_score(entities_filtradas)
                    
    except Exception:
        return {label: [] for label in result_labels}

    entities_by_label = {label: [] for label in result_labels}

    person_name_candidates: List[Dict[str, Any]] = []

    for entity in entities:
        raw_text = entity["text"]
        text_norm = raw_text.lower().strip()
        label = entity["label"]
        resolved_group = entity.get("_group")

        if len(text_norm) <= 2:
            continue
        # Remove línguas do lado GLiNER.
        # As línguas devem vir apenas do regex extractor.
        if _is_language_entity(raw_text):
            continue
        
        # rejeita entidades com mais de 60% de números ou símbolos,
        # exceto casos permitidos como "12º ano", "2D" e "3D"
        if _has_more_than_60_percent_numbers_or_symbols(raw_text):
            continue

        # remove genéricos
        group_invalid = set()

        for group_name, group_labels in ENTITY_GROUPS.items():
            if label in group_labels:
                group_invalid = INVALID_ENTITIES.get(group_name, set())
                break

        if text_norm in group_invalid:
            continue
        if text_norm in GENERIC_INVALID:
            continue

        if resolved_group == "skills":
            normalized_skill = (
                _normalize_if_in_vocab(raw_text, SOFTWARE_NORMALIZATION_INDEX)
                or _normalize_if_in_vocab(raw_text, PROGRAMMING_NORMALIZATION_INDEX)
                or _normalize_if_in_vocab(raw_text, GENERAL_SKILLS_NORMALIZATION_INDEX)
                or raw_text
            )

            if _is_valid_skill(normalized_skill):
                entities_by_label["skills"].append(normalized_skill)
            continue

        # address
        if resolved_group == "address":
            expanded = _expand_address_smart(text, raw_text)
            entities_by_label["address"].append(expanded)
            continue
      
        # other direct labels
        if resolved_group == "person_name":
            person_name_candidates.append({
                "text": raw_text,
                "score": float(entity.get("_score", entity.get("score", 0.0))),
                "start": int(entity.get("start", 10**9)),
            })
            continue

        if resolved_group == "job_title":
            normalized_job_title = _normalize_if_in_vocab(raw_text, JOB_TITLE_NORMALIZATION_INDEX) or raw_text
            entities_by_label["job_title"].append(normalized_job_title)
            continue

        if resolved_group in {"company", "university", "degree"}:
            entities_by_label[resolved_group].append(raw_text)
            continue
    
    # Escolher apenas o melhor nome de pessoa
    entities_by_label["person_name"] = _choose_best_person_name(person_name_candidates)

    # --------------------------------------------------
    # Optional regex fallbacks
    # --------------------------------------------------
    if use_regex_fallbacks:
        entities_by_label = _apply_regex_fallbacks_to_entities(
            text=text,
            entities_by_label=entities_by_label,
            use_job_title_regex_fallback=use_job_title_regex_fallback,
            use_skill_regex_fallback=use_skill_regex_fallback,
            use_software_regex_fallback=use_software_regex_fallback,
            use_programming_regex_fallback=use_programming_regex_fallback,
            use_general_skills_regex_fallback=use_general_skills_regex_fallback,
        )

    return entities_by_label

def process_texts_gliner(
    texts: Union[List[str], List[Dict[str, str]], Dict[str, Any]],
    model: Optional[GLiNER] = None,
    gliner_threshold: float = 0.15,
    debug_save_json: bool = False,
    debug_output_path: Optional[Path | str] = OUTPUTS_DIR,
    use_regex_fallbacks: bool = USE_REGEX_FALLBACKS,
    use_job_title_regex_fallback: bool = USE_JOB_TITLE_REGEX_FALLBACK,
    use_skill_regex_fallback: bool = USE_SKILL_REGEX_FALLBACK,
    use_software_regex_fallback: bool = USE_SOFTWARE_REGEX_FALLBACK,
    use_programming_regex_fallback: bool = USE_PROGRAMMING_REGEX_FALLBACK,
    use_general_skills_regex_fallback: bool = USE_GENERAL_SKILLS_REGEX_FALLBACK,
    use_regex_fallbacks_on_chunks: bool = USE_REGEX_FALLBACKS_ON_CHUNKS,
) -> pd.DataFrame:

    if model is None:
        model = get_gliner_model()

    if isinstance(texts, list):
        text_items = list(enumerate(texts))
    elif isinstance(texts, dict):
        if any(k in texts for k in ("gliner_text", "raw_text", "text")):
            text_items = [(0, texts)]
        else:
            text_items = list(texts.items())
    else:
        raise ValueError("texts must be a list or dict")

    rows = []

    for _, item in text_items:
        try:
            if isinstance(item, dict):
                gliner_text = item.get("gliner_text", "") or item.get("text", "")
                raw_text = item.get("raw_text", "") or gliner_text
            else:
                gliner_text = item
                raw_text = item

            entities_dict = _extrair_entidades_texto_longo(
                gliner_text,
                model=model,
                threshold=gliner_threshold,
                debug_save_json=debug_save_json,
                debug_output_path=debug_output_path,
                use_regex_fallbacks=use_regex_fallbacks,
                use_job_title_regex_fallback=use_job_title_regex_fallback,
                use_skill_regex_fallback=use_skill_regex_fallback,
                use_software_regex_fallback=use_software_regex_fallback,
                use_programming_regex_fallback=use_programming_regex_fallback,
                use_general_skills_regex_fallback=use_general_skills_regex_fallback,
                use_regex_fallbacks_on_chunks=use_regex_fallbacks_on_chunks,
            )

            row: Dict[str, Any] = {}

            for label in [
                "person_name",
                "job_title",
                "company",
                "university",
                "degree",
                "address",
                "skills"
            ]:
            
                entities = entities_dict.get(label, [])

                clean_entities = []
                for e in entities:
                    if not isinstance(e, str):
                        continue

                    e_clean = " ".join(e.replace("\n", " ").replace("\r", " ").split())
                    if e_clean:
                        clean_entities.append(e_clean.strip())

                if label == "job_title":
                    clean_entities = [
                        _normalize_if_in_vocab(e, JOB_TITLE_NORMALIZATION_INDEX) or e
                        for e in clean_entities
                    ]

                if label == "skills":
                    normalized_skills = []
                    for e in clean_entities:
                        normalized = (
                            _normalize_if_in_vocab(e, SOFTWARE_NORMALIZATION_INDEX)
                            or _normalize_if_in_vocab(e, PROGRAMMING_NORMALIZATION_INDEX)
                            or _normalize_if_in_vocab(e, GENERAL_SKILLS_NORMALIZATION_INDEX)
                            or e
                        )
                        normalized_skills.append(normalized)

                    clean_entities = normalized_skills

                if label == "person_name":
                    heuristic_name = extract_top_name_candidate(raw_text)

                    if not clean_entities and heuristic_name:
                        clean_entities = [heuristic_name]

                if label == "address":
                    clean_entities = _choose_first_address(clean_entities)

                if label == "job_title":
                    clean_entities = [
                        e for e in clean_entities
                        if _is_valid_job_title(e)
                    ]

                if label in {"company", "university","degree"}:
                    clean_entities = _remove_substring_duplicates(clean_entities)

                # Remove duplicates ignoring case
                unique = {}
                for e in clean_entities:
                    key = e.lower()
                    if key not in unique:
                        unique[key] = e

                clean_entities = list(unique.values())

                row[label] = clean_entities
            
            # Normalize degrees
            row["degree"] = _normalize_degree_entries(row.get("degree") or [])

            # Use the default degree
            row["degree"] = _ensure_default_degree(
                row.get("university") or [],
                row.get("degree") or [])
            
            row["experience_years"] = extract_experience_years(
                raw_text=raw_text,
                degrees=row.get("degree") or [],
                universities=row.get("university") or [],
                job_titles=row.get("job_title") or [],
            ) or 0

            # Convert lists to string
            for key in row:
                if isinstance(row[key], list):
                    row[key] = ", ".join(row[key]) if row[key] else None


            rows.append(row)

        except Exception as e:
            rows.append({"error": str(e)})

    return pd.DataFrame(rows)

