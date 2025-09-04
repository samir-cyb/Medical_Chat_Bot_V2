import json
import os
from .config import KNOWLEDGE_BASE_PATH

# Symptom weights for more accurate matching - EXPANDED
symptom_weights = {
    # High urgency symptoms
    "sharp pain lower right abdomen": 9, "right lower quadrant pain": 9, "rebound tenderness": 10,
    "chest pain": 7, "pain radiating to arm or jaw": 9, "shortness of breath": 8, "heart palpitations": 9,
    "difficulty breathing": 9, "severe dehydration": 9, "bleeding gums": 8, "nosebleeds": 8, 
    "severe abdominal pain": 9, "persistent vomiting": 9, "lethargy": 8, "restlessness": 7,
    "sudden drop in temperature": 9, "confusion": 9, "very high fever": 8, "severe bleeding": 10,
    "stiff neck": 8, "light sensitivity": 7, "severe dizziness": 8, "fainting": 9,
    # New high urgency symptoms
    "sudden chest tightness": 9, "severe shortness of breath": 9, "uncontrolled bleeding": 10,
    "sudden severe headache": 9, "loss of consciousness": 10, "rapid heartbeat": 8,
    "severe allergic reaction": 9, "anaphylaxis": 10, "seizures": 10, "sudden vision loss": 9,
    
    # Dengue specific symptoms
    "high fever": 8, "severe headache": 7, "pain behind eyes": 8, "severe muscle pain": 7,
    "severe joint pain": 7, "skin rash": 6, "mild bleeding": 7, "low platelet count": 8,
    "easy bruising": 7, "petechiae": 8, "red spots on skin": 6, "increased thirst": 5,
    # New dengue-specific symptoms
    "abdominal tenderness": 7, "persistent nausea": 6, "mucous membrane bleeding": 8,
    "low white blood cell count": 7, "severe fatigue with fever": 7, "sudden fever onset": 8,
    
    # Chikungunya specific symptoms
    "high fever with abrupt onset": 7, "debilitating joint pain": 9, "joint swelling": 7, "maculopapular rash": 6,
    "conjunctival injection": 6, "nausea": 5, "fatigue": 5, "severe joint stiffness": 7,
    "symmetric joint involvement": 6, "joint redness": 5, "morning joint stiffness": 6,
    # New chikungunya-specific symptoms
    "persistent joint discomfort": 7, "bilateral joint pain": 6, "prolonged joint stiffness": 7,
    "severe myalgia": 7, "red swollen joints": 6, "chronic joint pain": 6,
    
    # Medium urgency symptoms
    "high fever": 7, "severe body aches": 7, "persistent cough": 6, "extreme fatigue": 6,
    "widespread rash": 6, "eye pain": 6, "nausea": 6, "vomiting": 6, "diarrhea": 5,
    "loss of appetite": 4, "swollen lymph nodes": 6, "ear pain": 5, "sinus pressure": 4,
    # New medium urgency symptoms
    "moderate abdominal pain": 6, "recurrent vomiting": 6, "persistent diarrhea": 6,
    "mild shortness of breath": 6, "chest discomfort": 6, "moderate headache": 5,
    "frequent nosebleeds": 6, "mild dehydration": 6, "persistent sore throat": 5,
    "muscle cramps": 5, "joint discomfort": 5, "skin itching with rash": 5,
    
    # Low urgency symptoms
    "mild fever": 4, "slight runny nose": 2, "occasional sneezing": 2, "mild headache": 3,
    "joint pain": 5, "rash": 5, "mild muscle pain": 3, "runny nose": 3, "body aches": 4,
    "cough": 4, "sore throat": 4, "changing mole": 5, "new skin growth": 5, "severe acne": 4,
    "headache": 4, "migraine": 5, "head pain": 4, "throbbing head": 4, "mild fatigue": 3,
    "occasional cough": 3, "tiredness": 3, "mild sore throat": 3,"System: congestion": 3,
    
    # New low urgency symptoms
    "slight cough": 2, "mild nasal congestion": 2, "occasional headache": 3,
    "slight sore throat": 3, "mild body aches": 3, "slight fatigue": 3,
    "mild nausea": 4, "occasional diarrhea": 4, "mild rash": 4,
    "slight joint pain": 4, "mild skin irritation": 3, "occasional dizziness": 4,
    "mild eye irritation": 3, "slight muscle soreness": 3, "mild thirst": 3
}

# Symptom mapping for normalization - EXPANDED
symptom_mapping = {
    "fever": ["fever", "temperature", "hot", "burning up", "have a temp", "running a fever", "feeling feverish", "got a fever", "having a fever", "body temp high", "fevers", "febrile", "feeling warm all over", "shaking with fever", "temperature's up", "chills with fever", "feverish chills", "hot flashes", "feeling overheated", "body's burning", "temperature spiking", "fever coming on", "really warm", "flushed with fever", "hot and sweaty"],
    "high fever": ["persistent high fever", "high fever", "very hot", "102 fever", "103 fever", "104 fever", "105 fever", "spiking fever", "very high temperature", "fever over 102", "extremely high fever", "dangerous fever", "fever spiking high", "temperature soaring", "intense fever", "severe temperature rise", "skyrocketing fever", "blazing fever", "really high temp", "fever out of control", "temperature through the roof", "scorching fever", "fever way too high", "boiling up", "extreme temperature", "high-grade fever"],
    "mild fever": ["mild fever", "slight fever", "low grade fever", "99 fever", "100 fever", "101 fever", "low-grade temperature", "low fever", "just a bit feverish", "mild temperature", "slight temp", "not too high fever", "low-level fever", "gentle fever", "subtle fever", "just warm", "barely a fever", "touch of fever", "mildly warm", "slight temperature increase", "low-key fever", "not quite a fever", "warmish feeling", "faint fever", "borderline fever"],
    "headache": ["headache", "head pain", "head hurting", "head ache", "head throbbing", "pounding head", "aching head", "head hurts", "splitting headache", "pressure in head", "migraine-like", "throbbing temples", "pain in my head", "head is killing me", "dull headache", "skull pain", "head pounding", "tension headache", "ache in head", "head discomfort", "pulsing headache", "brain hurts", "head feels heavy", "stabbing head pain", "constant head ache"],
    "severe headache": ["severe headache", "bad headache", "worst headache", "excruciating headache", "debilitating headache", "unbearable headache", "intense headache", "crushing headache", "headache from hell", "severe head pain", "agonizing headache", "extreme head ache", "throbbing severely", "can't stand the headache", "debilitating head pain", "killer headache", "head-splitting pain", "vicious headache", "brutal head pain", "relentless headache", "excruciating head pressure", "head pain unbearable", "severe throbbing head", "worst head pain ever", "crippling headache"],
    "mild headache": ["mild headache", "slight headache", "minor headache", "small headache", "light headache", "gentle head pain", "not too bad headache", "subtle head ache", "mild head hurting", "faint headache", "low-level headache", "barely a headache", "slight pressure in head", "minor head discomfort", "easy headache", "just a little headache", "mild head tension", "slight head soreness", "barely noticeable headache", "low-grade head pain", "mild head irritation", "slight throbbing", "gentle head ache", "minor head ache", "faint head pain"],
    "cough": ["cough", "coughing", "hacking", "hacking cough", "coughing fit", "dry cough", "wet cough", "coughing up", "persistent hack", "cough attack", "whooping cough", "barking cough", "coughing spell", "raspy cough", "irritating cough", "chesty cough", "tickly cough", "cough won't stop", "rough cough", "deep cough", "hacking up a lung", "coughing hard", "throaty cough", "cough spasm", "irritated cough"],
    "persistent cough": ["persistent cough", "constant cough", "nonstop coughing", "continuous cough", "won't stop coughing", "ongoing cough", "lingering cough", "chronic coughing", "cough that won't go away", "endless cough", "repeated coughing", "cough all the time", "unrelenting cough", "cough day and night", "prolonged cough", "cough keeps going", "never-ending cough", "relentless coughing", "cough won't quit", "ongoing hacking", "stubborn cough", "coughing constantly", "persistent hack", "cough round the clock", "chronic cough"],
    "runny nose": ["runny nose", "nose running", "rhinorrhea", "dripping nose", "nasal discharge", "stuffy runny nose", "nose dripping", "snotty nose", "constant sniffles", "leaky nose", "runny nostrils", "nasal drip", "nose won't stop running", "watery nose", "sniffling", "snot running", "nose like a faucet", "drippy nose", "constant nose drip", "wet nose", "runny sinuses", "nasal leakage", "sneezing and runny", "flowing nose", "mucus dripping"],
    "sore throat": ["sore throat", "throat pain", "throat hurting", "painful throat", "scratchy throat", "irritated throat", "raw throat", "throat ache", "hurts to swallow", "throat irritation", "sore in throat", "throat on fire", "itchy throat", "tender throat", "inflamed throat", "scratchy sore throat", "throat feels raw", "pain when swallowing", "throat burning", "sore and scratchy", "irritated throat lining", "throat soreness", "hurts to talk", "rough throat", "swollen throat feeling"],
    "body aches": ["body aches", "muscle aches", "aching body", "body pains", "generalized ache", "whole body hurts", "all over aches", "body soreness", "aching all over", "general body pain", "full body ache", "widespread aches", "body feeling sore", "aches everywhere", "sore body", "total body soreness", "all-over body pain", "general achiness", "body feels beat up", "whole body aching", "sore all over", "generalized soreness", "body hurts everywhere", "diffuse body pain", "aching limbs and body"],
    "muscle pain": ["muscle pain", "myalgia", "muscles hurting", "sore muscles", "muscle soreness", "aching muscles", "muscle aches", "pain in muscles", "tender muscles", "muscle discomfort", "cramping muscles", "stiff muscles", "hurting muscles", "muscle tenderness", "deep muscle pain", "muscle cramps", "sore muscle tissue", "aching in muscles", "muscle stiffness", "painful muscles", "muscles feel sore", "muscle tightness", "deep muscle aches", "muscle throbbing", "sore and tight muscles"],
    "joint pain": ["joint pain", "aching joints", "joints hurting", "sore joints", "arthralgia", "joint discomfort", "pain in joints", "joint aches", "stiff joints", "hurting joints", "joint soreness", "inflamed joints", "tender joints", "joint stiffness", "achy joints", "painful joint movement", "joints ache bad", "sore joint areas", "joint tenderness", "aching in joints", "joint pain and stiffness", "hurts to move joints", "joint irritation", "stiff and sore joints", "joint achiness"],
    "severe joint pain": ["severe joint pain", "bad joint pain", "joints very painful", "excruciating joint pain", "debilitating joint pain", "intense joint pain", "extreme joint aches", "unbearable joint pain", "severe arthralgia", "agonizing joints", "crippling joint pain", "joints killing me", "severe joint discomfort", "debilitating joint aches", "worst joint pain", "vicious joint pain", "relentless joint pain", "excruciating joint aches", "joints in agony", "severe joint soreness", "intense joint agony", "unbearable joint stiffness", "crippling joint discomfort", "extreme joint throbbing", "joints hurt terribly"],
    "rash": ["rash", "skin rash", "red spots", "skin irritation", "skin eruption", "skin redness", "itchy rash", "red rash", "bumpy skin", "skin breakout", "rashes", "allergic rash", "red patches", "skin inflammation", "eruption on skin", "itchy red spots", "skin flare-up", "red irritated skin", "rashy skin", "skin blotches", "irritated skin patches", "red skin bumps", "skin breaking out", "itchy skin rash", "red skin eruption"],
    "maculopapular rash": ["maculopapular rash", "bumpy rash", "raised rash", "rash with bumps", "papular rash", "flat and raised rash", "macules and papules", "spotty bumpy rash", "raised red spots", "bumpy red rash", "papule rash", "macular papular", "rough rash", "textured rash", "elevated rash", "bumpy red patches", "mixed rash", "raised and flat rash", "papules and macules", "bumpy skin eruption", "red bumpy spots", "raised skin rash", "lumpy rash", "maculopapular spots", "textured skin rash"],
    "widespread rash": ["widespread rash", "rash all over", "body rash", "rash covering body", "extensive rash", "full body rash", "rash everywhere", "all over body rash", "generalized rash", "rash on whole body", "diffuse rash", "body-wide rash", "extensive skin rash", "rash spreading all over", "total body rash", "rash across body", "everywhere rash", "whole-body skin rash", "rash all over skin", "widespread skin eruption", "body covered in rash", "universal rash", "rash on entire body", "total skin rash", "generalized skin breakout"],
    "nausea": ["nausea", "queasy", "feeling sick", "sick to stomach", "upset stomach", "feel like vomiting", "nauseous", "stomach queasy", "feeling nauseated", "woozy", "stomach upset", "gonna throw up", "sick feeling", "queasiness", "nausea feeling", "feeling queasy", "stomach churning", "nauseated", "sick to my stomach", "feeling off", "stomach feels bad", "queasy stomach", "about to puke", "nausea coming on", "feeling woozy"],
    "vomiting": ["vomiting", "throwing up", "puking", "emesis", "regurgitating", "barfing", "heaving", "spewing", "upchucking", "vomiting episode", "being sick", "tossing cookies", "retching", "vomit", "projectile vomiting", "throwing up a lot", "puking my guts out", "vomiting hard", "sick vomiting", "gagging and vomiting", "barfing up", "vomiting spells", "chucking up", "vomiting violently", "heaving up"],
    "persistent vomiting": ["persistent vomiting", "constant vomiting", "can't stop vomiting", "repeated vomiting", "continuous vomiting", "ongoing vomiting", "nonstop puking", "frequent vomiting", "vomiting repeatedly", "uncontrollable vomiting", "vomiting all the time", "lingering vomiting", "chronic vomiting", "vomiting won't stop", "endless throwing up", "vomiting over and over", "relentless puking", "constant throwing up", "vomiting nonstop", "persistent puking", "can't quit vomiting", "throwing up constantly", "ongoing puking", "vomiting keeps going", "unstoppable vomiting"],
    "abdominal pain": ["abdominal pain", "stomach pain", "belly pain", "tummy ache", "abdominal discomfort", "gut pain", "stomach ache", "pain in abdomen", "belly hurts", "cramping in stomach", "abdominal cramps", "tummy pain", "stomach hurting", "pain in belly", "abdominal ache", "gut ache", "stomach cramps", "belly discomfort", "tummy hurting", "abdominal soreness", "pain in my stomach", "crabby stomach", "aching belly", "stomach twinges", "gut discomfort"],
    "severe abdominal pain": ["severe abdominal pain", "bad stomach pain", "extreme abdominal pain", "excruciating stomach pain", "intense belly pain", "severe stomach ache", "agonizing abdominal pain", "debilitating gut pain", "unbearable stomach pain", "severe cramps", "worst abdominal pain", "crushing stomach pain", "extreme tummy ache", "severe abdominal discomfort", "intense abdominal cramps", "killer stomach pain", "vicious abdominal pain", "relentless gut pain", "excruciating belly pain", "severe stomach cramps", "agonizing tummy pain", "unbearable abdominal ache", "crippling stomach pain", "extreme gut ache", "stomach pain from hell"],
    "shortness of breath": ["shortness of breath", "breathing difficulty", "can't catch breath", "labored breathing", "breathlessness", "hard to breathe", "out of breath", "gasping for air", "difficulty breathing", "wheezing", "tight chest breathing", "struggling to breathe", "short winded", "breath coming short", "dyspnea", "can't get enough air", "breathing feels tight", "short of air", "panting", "breathing hard", "gasping", "lungs feel tight", "trouble getting air", "breath catching", "hard to catch breath"],
    "chest pain": ["chest pain", "chest hurting", "pain in chest", "chest discomfort", "tightness in chest", "chest ache", "pain in the chest", "chest pressure", "sharp chest pain", "dull chest pain", "chest tightness", "hurting in chest", "chest soreness", "aching chest", "stabbing chest pain", "chest feels heavy", "pain across chest", "tight chest pain", "chest throbbing", "sore chest", "burning in chest", "chest pain on breathing", "sharp pain in chest", "constant chest ache", "chest wall pain"],
    "bleeding gums": ["bleeding gums", "gums bleeding", "blood when brushing", "blood from gums", "gingival bleeding", "gums bleed easily", "bleeding when flossing", "blood in spit", "gums hemorrhaging", "easy gum bleeding", "gums leaking blood", "blood on toothbrush", "swollen bleeding gums", "gum bleed", "oral bleeding", "gums oozing blood", "bleeding when chewing", "gums bleed when brushing", "constant gum bleeding", "blood from mouth", "gums bleeding a lot", "red gums bleeding", "sore and bleeding gums", "gums leaking", "bleeding oral tissue"],
    "nosebleeds": ["nosebleeds", "nose bleeding", "blood from nose", "epistaxis", "nasal bleeding", "nose bleed", "bleeding nose", "frequent Nosebleeds", "blood dripping from nose", "nose hemorrhaging", "uncontrolled nosebleed", "nose gushing blood", "recurrent nosebleeds", "nose starting to bleed", "bloody nose", "nosebleed episodes", "constant nose bleeding", "nosebleeds all the time", "blood pouring from nose", "frequent nasal bleeding", "nosebleed won't stop", "heavy nosebleed", "nasal hemorrhage", "nose bleeding heavily", "persistent nosebleeds"],
    "lethargy": ["lethargy", "extremely tired", "no energy", "profound fatigue", "complete exhaustion", "can't get out of bed", "feeling drained", "utterly exhausted", "worn out", "listless", "sluggish", "fatigued", "zero energy", "totally wiped out", "overwhelming tiredness", "bone-tired", "dead tired", "no strength", "completely worn out", "exhausted all the time", "feeling run down", "total lack of energy", "can barely move", "sapped energy", "extreme weariness"],
    "confusion": ["confusion", "disorientation", "can't think clearly", "mental fog", "altered mental state", "brain fog", "confused state", "disoriented", "fuzzy thinking", "mental confusion", "not thinking straight", "dazed", "bewildered", "mixed up", "cloudy mind", "foggy brain", "can't focus", "mind feels off", "scattered thoughts", "feeling lost", "mental haze", "disoriented thinking", "brain not working", "trouble concentrating", "mind all over the place"],
    "pain behind eyes": ["pain behind eyes", "eye socket pain", "retro-orbital pain", "pain behind eyeballs", "deep eye pain", "aching behind eyes", "pressure behind eyes", "hurting behind eyes", "orbital pain", "eyeball ache", "pain in eye sockets", "behind the eyes pain", "deep ocular pain", "retroorbital ache", "eyes hurting deep", "eye socket soreness", "pain deep in eyes", "aching eye sockets", "pressure in eyeballs", "hurting behind eyeballs", "retro-orbital discomfort", "deep pain behind eyes", "eye pain deep inside", "sore behind eyes", "throbbing behind eyes"],
    "conjunctival injection": ["conjunctival injection", "red eyes", "bloodshot eyes", "eye redness", "inflamed eyes", "reddened eyes", "eyes look bloodshot", "pink eyes", "irritated red eyes", "vascular eyes", "conjunctiva red", "eyes injected", "red sclera", "blood vessels in eyes", "eyes turning red", "red irritated eyes", "eyes look inflamed", "sore red eyes", "bloodshot and irritated", "red eyeballs", "eyes burning red", "inflamed eye appearance", "red and sore eyes", "eyes super red", "conjunctival redness"],
    "joint swelling": ["joint swelling", "swollen joints", "puffy joints", "inflamed joints", "joint inflammation", "joints puffed up", "enlarged joints", "swelling in joints", "joint edema", "bulging joints", "inflamed and swollen joints", "joints getting bigger", "puffy and sore joints", "joint puffiness", "swollen and painful joints", "joints swollen up", "big swollen joints", "joint bloating", "swollen joint areas", "puffy joint swelling", "inflamed joint tissue", "joints looking swollen", "swollen and stiff joints", "joint enlargement", "painful joint swelling"],
    "easy bruising": ["easy bruising", "bruising easily", "unexplained bruises", "bruises without injury", "prone to bruising", "bruises appearing", "easy to bruise", "mysterious bruises", "bruising for no reason", "skin bruises easily", "frequent bruising", "unusual bruising", "bruises show up easily", "minimal trauma bruising", "spontaneous bruising", "bruises popping up", "easy skin bruising", "bruising with no cause", "random bruises", "bruises from nothing", "skin marks easily", "frequent unexplained bruises", "bruising too easily", "bruises all over", "sensitive to bruising"],
    "petechiae": ["petechiae", "pinpoint red spots", "small red dots", "red specks on skin", "tiny red spots", "petechial rash", "red pinpoints", "small hemorrhages", "red dots on skin", "speckled red spots", "pinhead red spots", "tiny blood spots", "red freckles", "small purple spots", "capillary breaks", "tiny red marks", "red speckles", "small blood dots", "petechial spots", "red pinpricks", "tiny hemorrhagic spots", "red skin specks", "small red blemishes", "pinpoint hemorrhages", "red dots under skin"],
    "stiff neck": ["stiff neck", "neck stiffness", "can't move neck", "neck rigidity", "sore stiff neck", "neck won't bend", "rigid neck", "neck pain and stiffness", "locked neck", "neck frozen", "difficulty turning neck", "neck muscle stiffness", "stiff and painful neck", "neck inflexibility", "tense neck", "neck won't move", "stiff neck muscles", "neck locked up", "hard to turn neck", "neck feels tight", "rigid and sore neck", "neck stuck", "painful neck stiffness", "neck won't rotate", "stiff neck and sore"],
    "light sensitivity": ["light sensitivity", "sensitive to light", "photophobia", "lights hurt eyes", "bright lights bother me", "eyes sensitive to light", "can't stand bright lights", "light intolerance", "glare sensitivity", "eyes hurt in light", "photophobic", "squinting in light", "light makes eyes ache", "aversion to light", "bright light pain", "eyes can't handle light", "light bothers eyes", "sensitive to bright light", "lights too bright", "eyes hurt with light", "glare hurts eyes", "light sensitivity in eyes", "bright light discomfort", "eyes squinting in light", "painful light exposure"],
    "loss of appetite": ["loss of appetite", "not hungry", "no desire to eat", "reduced appetite", "appetite gone", "no interest in food", "lack of hunger", "anorexia", "don't feel like eating", "appetite loss", "zero appetite", "food aversion", "not eating much", "decreased hunger", "poor appetite", "no urge to eat", "food doesn't appeal", "can't eat", "hunger gone", "not wanting food", "appetite really low", "no taste for food", "eating feels off", "no craving for food", "feeling full without eating"],
    "swollen lymph nodes": ["swollen lymph nodes", "swollen glands", "enlarged lymph nodes", "lumps in neck", "swollen nodes", "lymph node swelling", "glands puffed up", "tender swollen glands", "neck lumps", "enlarged glands", "swollen lymphs", "lymphadenopathy", "bulging lymph nodes", "swollen under jaw", "painful swollen nodes", "lumpy glands", "swollen neck glands", "big lymph nodes", "tender lymph nodes", "swollen lumps in neck", "glands feel big", "lymph node enlargement", "swollen and sore glands", "puffy lymph nodes", "neck glands swollen"],
    "dehydration": ["dehydration", "dehydrated", "dry mouth", "excessive thirst", "dark urine", "feeling thirsty all the time", "dry skin", "lack of fluids", "thirsty and dry", "dehydrated feeling", "not enough water", "sunken eyes", "dry lips", "thirstiness", "fluid loss", "parched", "really thirsty", "feeling dried out", "low hydration", "thirsty constantly", "dry and thirsty", "no moisture in mouth", "dehydration symptoms", "craving water", "body feels dry"]
}

# Dengue-prone areas in Bangladesh (2025)
dengue_prone_areas = [
    "Dhaka", "Chattogram", "Barishal", "Khulna", "Rajshahi", "Sylhet",
    "Gazipur", "Narayanganj", "Mymensingh", "Rangpur"
]

# Chikungunya-prone areas in Bangladesh (2025)
chikungunya_prone_areas = [
    "Dhaka", "Tejgaon", "Mohakhali"
]

# Malaria-prone areas in Bangladesh (2025)
malaria_prone_areas = [
    "Bandarban", "Rangamati", "Khagrachhari"
]

# This is our medical knowledge dataset.
medical_data = [
    {
        "symptoms": ["sharp pain lower right abdomen", "right lower quadrant pain", "abdominal pain with nausea", "rebound tenderness"],
        "specialty": "Gastroenterology / Emergency Medicine",
        "suggested_action": "Seek immediate medical attention at an Urgent Care or Emergency Room. This could indicate appendicitis, which is a medical emergency.",
        "urgency": "Urgent",
        "first_aid": "Do not eat or drink anything. Avoid taking pain medication. Apply a cold compress to the area.",
        "follow_up_questions": ["Is the pain getting worse?", "Do you have a fever?", "Is there any vomiting?"],
        "condition": "Appendicitis"
    },
    {
        "symptoms": ["chest pain", "pain radiating to arm or jaw", "shortness of breath", "heart palpitations"],
        "specialty": "Cardiology / Emergency Medicine",
        "suggested_action": "This is a medical emergency. Go to the nearest Emergency Room or call emergency services immediately. Do not drive yourself.",
        "urgency": "Emergency",
        "first_aid": "If you are with the person, have them sit down and rest. If they have prescribed nitroglycerin, help them take it. If they are unresponsive, call for emergency help.",
        "follow_up_questions": ["Is the pain sharp or dull?", "Are you feeling dizzy?", "Is there shortness of breath?"],
        "condition": "Heart Attack"
    },
    {
        "symptoms": ["persistent rash", "changing mole", "new skin growth", "severe acne"],
        "specialty": "Dermatology",
        "suggested_action": "Schedule a routine appointment with a dermatologist for evaluation.",
        "urgency": "Routine",
        "first_aid": "Keep the affected area clean and dry. Avoid scratching. Use over-the-counter hydrocortisone cream if it's an itch.",
        "follow_up_questions": ["Has the rash spread?", "Is it itchy?", "Is there any pain or pus?"],
        "condition": "Skin Condition"
    },
    {
        "symptoms": ["fever", "cough", "sore throat", "runny nose", "body aches"],
        "specialty": "Primary Care / General Practice",
        "suggested_action": "Schedule an appointment with your primary care doctor or visit an Urgent Care clinic for evaluation. Rest and hydrate.",
        "urgency": "Routine",
        "first_aid": "Drink plenty of fluids, get rest, and take over-the-counter fever reducers if needed. Use a saline nasal spray for congestion.",
        "follow_up_questions": ["How long have you had these symptoms?", "Is the fever high?", "Do you have a productive cough?"],
        "condition": "Common Cold or Flu"
    },
    {
        "symptoms": ["headache", "migraine", "head pain", "throbbing head"],
        "specialty": "Neurology / Primary Care",
        "suggested_action": "If the headache is severe or sudden, seek urgent care. For chronic headaches, schedule an appointment with a neurologist or your primary care doctor.",
        "urgency": "Varies",
        "first_aid": "Rest in a dark, quiet room. Apply a cold or warm compress to your head. Stay hydrated.",
        "follow_up_questions": ["Is this a new type of headache for you?", "Are you experiencing any vision changes?", "Is the pain on one side of your head?"],
        "condition": "Headache"
    },
    {
        "symptoms": ["mild fever", "slight runny nose", "occasional sneezing", "mild headache"],
        "specialty": "Primary Care / General Practice",
        "suggested_action": "Rest at home, drink plenty of fluids, and take over-the-counter cold medicine if needed. Monitor your symptoms.",
        "urgency": "Normal",
        "first_aid": "Get plenty of rest, drink warm fluids like tea with honey, use a humidifier, and take over-the-counter cold medicine.",
        "follow_up_questions": ["How long have you had these symptoms?", "Is your fever above 100.4°F (38°C)?", "Are you experiencing any body aches?"],
        "stage": "Normal Stage",
        "condition": "Common Cold"
    },
    {
        "symptoms": ["high fever", "severe body aches", "persistent cough", "extreme fatigue"],
        "specialty": "Primary Care / General Practice",
        "suggested_action": "Schedule an appointment with your doctor today. You may need antiviral medication if diagnosed early.",
        "urgency": "Intermediate",
        "first_aid": "Rest, stay hydrated, take fever reducers like acetaminophen or ibuprofen, and use a warm compress for body aches.",
        "follow_up_questions": ["How high is your fever?", "Are you having difficulty breathing?", "Do you have any underlying health conditions?"],
        "stage": "Intermediate Stage",
        "condition": "Influenza (Flu)"
    },
    {
        "symptoms": ["very high fever", "difficulty breathing", "chest pain", "severe dehydration", "confusion"],
        "specialty": "Emergency Medicine",
        "suggested_action": "Seek emergency medical care immediately. These symptoms could indicate a serious complication.",
        "urgency": "Dangerous",
        "first_aid": "While waiting for emergency care, try to keep the person hydrated with small sips of water and use cool compresses to reduce fever.",
        "follow_up_questions": ["When did the breathing difficulties start?", "Is the person able to speak in full sentences?", "What is their current temperature?"],
        "stage": "Dangerous Stage",
        "condition": "Severe Respiratory Infection"
    },
    {
        "symptoms": ["mild fever", "joint pain", "rash", "headache", "mild muscle pain"],
        "specialty": "Infectious Disease / Primary Care",
        "suggested_action": "Schedule an appointment with your doctor for evaluation. Rest and take acetaminophen for pain and fever.",
        "urgency": "Normal",
        "first_aid": "Rest, stay hydrated, take pain relievers for joint discomfort, and use calamine lotion for rash relief.",
        "follow_up_questions": ["When did the symptoms start?", "Have you traveled to areas with known Chikungunya outbreaks?", "Is the rash spreading?"],
        "stage": "Normal Stage",
        "condition": "Chikungunya Fever"
    },
    {
        "symptoms": ["high fever", "severe joint pain", "widespread rash", "eye pain", "nausea"],
        "specialty": "Infectious Disease",
        "suggested_action": "Seek medical attention within 24 hours. You may need specific testing and management.",
        "urgency": "Intermediate",
        "first_aid": "Rest in a cool environment, stay well-hydrated, take pain medication as directed, and use cool compresses for fever.",
        "follow_up_questions": ["How severe is the joint pain?", "Are you able to move your joints comfortably?", "Have you noticed any bleeding?"],
        "stage": "Intermediate Stage",
        "condition": "Chikungunya Fever"
    },
    {
        "symptoms": ["persistent high fever", "severe dehydration", "bleeding gums", "nosebleeds", "severe abdominal pain"],
        "specialty": "Emergency Medicine / Infectious Disease",
        "suggested_action": "Go to the emergency room immediately. This could indicate severe Chikungunya complications.",
        "urgency": "Dangerous",
        "first_aid": "While waiting for emergency care, keep the person lying down with elevated legs if showing signs of shock. Do not give aspirin or NSAIDs.",
        "follow_up_questions": ["Is there any visible bleeding?", "How long has the fever persisted?", "Is the person conscious and responsive?"],
        "stage": "Dangerous Stage",
        "condition": "Chikungunya Fever"
    },
    {
        "symptoms": ["mild fever", "headache", "mild body aches", "rash"],
        "specialty": "Infectious Disease / Primary Care",
        "suggested_action": "Schedule a doctor's appointment for evaluation. Monitor for any warning signs.",
        "urgency": "Normal",
        "first_aid": "Rest, drink plenty of fluids, take acetaminophen for fever and pain, and avoid mosquito bites to prevent spreading.",
        "follow_up_questions": ["When did the symptoms start?", "Have you been in areas with dengue outbreaks?", "Is the rash itchy?"],
        "stage": "Normal Stage",
        "condition": "Dengue Fever"
    },
    {
        "symptoms": ["high fever ", "severe headache", "pain behind eyes", "severe muscle pain", "vomiting"],
        "specialty": "Infectious Disease",
        "suggested_action": "Seek medical attention within 24 hours. You may need blood tests and close monitoring.",
        "urgency": "Intermediate",
        "first_aid": "Rest, stay hydrated with oral rehydration solutions, take acetaminophen for pain (avoid aspirin/NSAIDs), and use cool compresses.",
        "follow_up_questions": ["How many days have you had fever?", "Are you able to keep fluids down?", "Have you noticed any bleeding or bruising?"],
        "stage": "Intermediate Stage",
        "condition": "Dengue Fever"
    },
    {
        "symptoms": ["sudden drop in temperature", "severe abdominal pain", "persistent vomiting", "bleeding", "lethargy", "restlessness"],
        "specialty": "Emergency Medicine / Infectious Disease",
        "suggested_action": "This is a medical emergency. Go to the hospital immediately. This could indicate dengue hemorrhagic fever or dengue shock syndrome.",
        "urgency": "Dangerous",
        "first_aid": "While waiting for emergency care, keep the person hydrated with small sips of oral rehydration solution. Do not give aspirin or NSAIDs.",
        "follow_up_questions": ["Is there any visible bleeding?", "When was the last time they urinated?", "Are they conscious and responsive?"],
        "stage": "Dangerous Stage",
        "condition": "Dengue Fever"
    },
    {
        "symptoms": ["fever", "stiff neck", "headache", "confusion", "light sensitivity"],
        "specialty": "Emergency Medicine / Neurology",
        "suggested_action": "Seek emergency medical care immediately. These could be signs of meningitis.",
        "urgency": "Emergency",
        "first_aid": "Keep the person in a quiet, dark room. Do not give anything by mouth if they are confused or vomiting.",
        "follow_up_questions": ["When did the neck stiffness start?", "Is there a rash anywhere on the body?", "Any recent head injury?"],
        "condition": "Possible Meningitis"
    },
    {
        "symptoms": ["fever", "sore throat", "swollen lymph nodes", "fatigue", "loss of appetite"],
        "specialty": "Primary Care / Infectious Disease",
        "suggested_action": "Schedule an appointment with your doctor. This could be mononucleosis or another viral infection.",
        "urgency": "Routine",
        "first_aid": "Rest, hydrate, and use throat lozenges or salt water gargles for throat pain.",
        "follow_up_questions": ["How long have you felt fatigued?", "Is there any abdominal pain?", "Have you had recent contact with anyone who was sick?"],
        "condition": "Viral Infection"
    },
    
    {
    "symptoms": ["chest pain", "heartburn", "acid reflux", "bloating", "indigestion"],
    "specialty": "Gastroenterology",
    "suggested_action": "Schedule an appointment with a gastroenterologist. Avoid spicy foods and consider antacids.",
    "urgency": "Routine",
    "condition": "Acid Reflux / GERD"
},
{
    "symptoms": ["chest pain", "difficulty swallowing", "food getting stuck", "heartburn"],
    "specialty": "Gastroenterology", 
    "suggested_action": "Consult a gastroenterologist for an endoscopy evaluation.",
    "urgency": "Intermediate",
    "condition": "Esophageal Disorder"
},
{
    "symptoms": ["chest pain", "anxiety", "rapid heartbeat", "shortness of breath", "sweating"],
    "specialty": "Cardiology / Psychiatry",
    "suggested_action": "Consult both a cardiologist to rule out heart issues and a mental health professional for anxiety evaluation.",
    "urgency": "Intermediate",
    "condition": "Anxiety / Panic Attack"
}
    
    
]

# A separate database for doctor profiles
doctor_profiles = [
    {
        "name": "Dr. Aruna Reddy",
        "specialty": "Cardiology",
        "contact": "+91-9876543210",
        "location": "Apolo Hospital, Delhi"
    },
    {
        "name": "Dr. Sameer Khan",
        "specialty": "Gastroenterology",
        "contact": "+91-9988776655",
        "location": "Max Healthcare, Delhi"
    },
    {
        "name": "Dr. Priya Sharma",
        "specialty": "Dermatology",
        "contact": "+91-9012345678",
        "location": "Fortis Hospital, Mumbai"
    },
    {
        "name": "Dr. Rajiv Gupta",
        "specialty": "Neurology",
        "contact": "+91-9765432109",
        "location": "AIIMS, Delhi"
    },
    {
        "name": "Dr. Pooja Singh",
        "specialty": "General Practice",
        "contact": "+91-9123456789",
        "location": "City Clinic, Delhi"
    },
    {
        "name": "Dr. Vikram Malhotra",
        "specialty": "Infectious Disease",
        "contact": "+91-9234567890",
        "location": "Medanta Hospital, Delhi"
    },
    {
        "name": "Dr. Anjali Mehta",
        "specialty": "Emergency Medicine",
        "contact": "+91-9345678901",
        "location": "Fortis Hospital, Mumbai"
    },
    {
        "name": "Dr. Mohammad Rahman",
        "specialty": "Infectious Disease",
        "contact": "+880-1712345678",
        "location": "Dhaka Medical College Hospital, Dhaka"
    },
    {
        "name": "Dr. Fatima Begum",
        "specialty": "General Practice",
        "contact": "+880-1812345678",
        "location": "Popular Diagnostic Centre, Dhaka"
    },
    {
        "name": "Dr. Abdul Hamid",
        "specialty": "Emergency Medicine",
        "contact": "+880-1912345678",
        "location": "Ibn Sina Hospital, Dhaka"
    }
]

def create_knowledge_base():
    """
    Creates a JSON file containing the medical knowledge data.
    """
    data = {
        "medical_data": medical_data,
        "doctor_profiles": doctor_profiles,
        "symptom_weights": symptom_weights,
        "symptom_mapping": symptom_mapping,
        "dengue_prone_areas": dengue_prone_areas,
        "chikungunya_prone_areas": chikungunya_prone_areas,
        "malaria_prone_areas": malaria_prone_areas
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(KNOWLEDGE_BASE_PATH), exist_ok=True)
    
    with open(KNOWLEDGE_BASE_PATH, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ Knowledge base created successfully at '{KNOWLEDGE_BASE_PATH}' with {len(medical_data)} medical entries and {len(doctor_profiles)} doctor profiles.")

if __name__ == "__main__":
    create_knowledge_base()