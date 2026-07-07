import streamlit as st
import pandas as pd
import base64
import urllib.parse as _url
import plotly.express as px
import plotly.graph_objects as go
import datetime
from difflib import SequenceMatcher
import numpy as np
import re
import io
import os

# Optional fuzzy matching support
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

# Optional AG Grid support with Streamlit fallback.
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode
    _HAS_AGGRID = True
except Exception:
    _HAS_AGGRID = False

# ================= CONFIGURATION =================
st.set_page_config(
    page_title="CDH NEXUS Clinical OmniSuite | KPJ University",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_BASENAME  = "subspecialityprediction"
GEO_FILENAME   = "kpj_geo"
PROJECT_FILENAME = "milestone"
FUZZY_THRESHOLD = 65
DIGITAL_HEALTH_FILENAME = "cdhprojects"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= SESSION STATE INITIALISATION =================
# Initialised once at startup so show() and the sidebar can safely read
# these keys on the very first page load, before any login attempt.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "login_error" not in st.session_state:
    st.session_state.login_error = False

# --- KPJU THEME ---
COLOR_BG      = "#F6F2EC"   # Application background - Light Warm Beige (Executive Design System)
COLOR_SIDEBAR = "#F4EFE7"   # Sidebar background - soft warm neutral (Executive Design System)
COLOR_PRIMARY = "#8C7C68"   
COLOR_TITLE   = "#6D6256"   
COLOR_HOVER   = "#595046"   
COLOR_TEXT    = "#5D5348"   
COLOR_CARD    = "#FFFFFF"
COLOR_SECONDARY = "#013D54"


# ================= LOGIC: FELLOWSHIP RULES =================
FELLOWSHIP_RULES = {
    "Interventional Cardiology": ["pci", "angioplasty", "stent", "cto", "chronic total occlusion", "coronary", "angiogram", "radial", "scaffold", "thrombectomy", "intervention", "endovascular", "peripheral vascular", "complex pci"],
    "Non-Invasive & Geriatric Cardiology": ["non invasive", "stress test", "treadmill", "holter", "ecg", "diagnostic", "monitoring", "ambulatory", "geriatric", "elderly", "general cardiology"],
    "Cardiac Electrophysiology": ["electrophysiology", "pacemaker", "icd", "crt", "ablation", "arrhythmia", "atrial fibrillation", "device therapy", "pacing", "ep study", "device follow-up", "remote monitoring"],
    "Heart Failure & Critical Care": ["heart failure", "cardiomyopathy", "lvad", "transplant", "ejection fraction", "pump failure", "amyloidosis", "mechanical support", "mcs", "critical care", "intensive care", "cic", "shock"],
    "Structural Heart & Valvular": ["tavi", "mitraclip", "valvular", "aortic stenosis", "mitral regurgitation", "structural", "valve", "tricuspid", "laa closure"],
    "Cardiac Imaging": ["echocardiography", "ctca", "cmr", "mri", "strain imaging", "nuclear", "calcium score", "pet scan", "cardiac ct", "multimodality"],
    "Cardiovascular Genetics": ["genetics", "inherited", "dna", "familial", "genome", "hypertrophic cardiomyopathy", "hcm", "channelopathy", "brugada", "long qt", "genetic counseling"],
    "Preventive & Metabolic Medicine": ["lipid", "cholesterol", "statin", "preventive", "risk factor", "hypertension", "diabetes", "metabolic", "obesity", "smoking", "cessation", "cardiovascular risk", "adherence", "secondary prevention", "cardio-renal", "metabolic syndrome", "post-mi"],
    "Pulmonary Hypertension": ["pulmonary hypertension", "right heart", "pah", "embolism", "ph"],
    "Sports Cardiology": ["athlete", "sports", "endurance", "exercise physiology", "screening", "sudden death"],
    "Cardio-Oncology": ["cardiotoxicity", "cancer", "chemotherapy", "anthracycline", "oncology", "radiation", "immunotherapy"],
    "Cardio-Obstetrics": ["pregnancy", "maternal", "foetal", "pregnant", "peripartum"],
    "Paediatric & ACHD": ["paediatric", "congenital", "child", "infant", "tetralogy", "asd", "vsd", "blue baby", "achd", "adult congenital"]
}

# ================= GEOSPATIAL LOOKUP =================

@st.cache_data
def load_geo_lookup():
    """Load hospital and region coordinates from kpj_geo.csv."""
    try:
        geo_df = pd.read_csv(f"{GEO_FILENAME}.csv", sep=None, engine="python")
        geo_df.columns = [c.strip().lower() for c in geo_df.columns]

        rename_map = {}
        for col in geo_df.columns:
            if "hosp" in col: rename_map[col] = "Hospital_Name"
            elif "lat" in col: rename_map[col] = "Latitude"
            elif "long" in col or "lng" in col: rename_map[col] = "Longitude"
            elif "region" in col: rename_map[col] = "Region"
        geo_df.rename(columns=rename_map, inplace=True)

        required = {"Hospital_Name", "Latitude", "Longitude"}
        if not required.issubset(geo_df.columns):
            return {}, {}, False

        geo_df["Latitude"] = pd.to_numeric(geo_df["Latitude"], errors="coerce")
        geo_df["Longitude"] = pd.to_numeric(geo_df["Longitude"], errors="coerce")
        geo_df = geo_df.dropna(subset=["Latitude", "Longitude"])

        hospital_coords = {
            str(row["Hospital_Name"]).strip().lower(): [row["Latitude"], row["Longitude"]]
            for _, row in geo_df.iterrows()
        }

        region_coords = {}
        if "Region" in geo_df.columns:
            region_means = geo_df.groupby("Region")[["Latitude", "Longitude"]].mean()
            region_coords = {
                region: [r["Latitude"], r["Longitude"]]
                for region, r in region_means.iterrows()
            }

        return hospital_coords, region_coords, True
    except Exception:
        return {}, {}, False

# ================= LOGIC FUNCTIONS =================
def enrich_data_with_coords(df):
    """
    Assigns Latitude/Longitude to each row using coordinates loaded from
    kpj_geo.csv (via load_geo_lookup) instead of hardcoded Python dicts.
    Adds a 'Location_Confidence' column so the map/UI can be honest about
    which points are exact hospital matches vs. estimated region centroids,
    instead of silently faking precision for unmatched rows.
    """
    if "Latitude" not in df.columns: df["Latitude"] = None
    if "Longitude" not in df.columns: df["Longitude"] = None
    if "Location_Confidence" not in df.columns: df["Location_Confidence"] = None

    hospital_coords, region_coords, geo_file_found = load_geo_lookup()

    if not geo_file_found:
        st.warning(
            f"Could not find or read **{GEO_FILENAME}.csv** in the app folder. "
            "Geospatial coordinates cannot be plotted accurately until this file is added "
            "(expected columns: Hospital_Name, Latitude, Longitude, Region)."
        )
        df["Location_Confidence"] = "No Geo File"
        return df

    rng = np.random.default_rng(42)  # deterministic small jitter, not row-index-based
    fallback_region_coords = next(iter(region_coords.values()), [3.1390, 101.6869])  # last resort if even region is unknown

    for i, row in df.iterrows():
        if pd.isna(row["Latitude"]):
            match_found = False
            hosp_name = str(row.get("Hospital", "")).strip().lower()
            region_name = str(row.get("Region", "")).strip()

            # Exact hospital match
            if hosp_name in hospital_coords:
                coords = hospital_coords[hosp_name]
                jitter = rng.uniform(-0.003, 0.003, size=2)
                df.at[i, "Latitude"] = coords[0] + jitter[0]
                df.at[i, "Longitude"] = coords[1] + jitter[1]
                df.at[i, "Location_Confidence"] = "Exact"
                match_found = True

            # Partial hospital match
            if not match_found:
                for key, coords in hospital_coords.items():
                    if key in hosp_name or hosp_name in key:
                        jitter = rng.uniform(-0.003, 0.003, size=2)
                        df.at[i, "Latitude"] = coords[0] + jitter[0]
                        df.at[i, "Longitude"] = coords[1] + jitter[1]
                        df.at[i, "Location_Confidence"] = "Exact"
                        match_found = True
                        break

            # Region centroid fallback
            if not match_found and region_name in region_coords:
                coords = region_coords[region_name]
                jitter = rng.uniform(-0.05, 0.05, size=2)
                df.at[i, "Latitude"] = coords[0] + jitter[0]
                df.at[i, "Longitude"] = coords[1] + jitter[1]
                df.at[i, "Location_Confidence"] = "Estimated (Region)"
                match_found = True

            # Last-resort fallback
            if not match_found:
                jitter = rng.uniform(-0.05, 0.05, size=2)
                df.at[i, "Latitude"] = fallback_region_coords[0] + jitter[0]
                df.at[i, "Longitude"] = fallback_region_coords[1] + jitter[1]
                df.at[i, "Location_Confidence"] = "Unknown (Defaulted)"
    return df

def get_hospital_saturation(df):
    counts = {k: 0 for k in FELLOWSHIP_RULES.keys()}
    existing_subs = df["Subspecialty/Fellowship"].dropna().astype(str).tolist()
    for existing in existing_subs:
        existing_lower = existing.lower()
        matched = False
        for category in FELLOWSHIP_RULES.keys():
            if category.split(" & ")[0].lower() in existing_lower:
                counts[category] += 1
                matched = True
                break
        if not matched:
            for category, keywords in FELLOWSHIP_RULES.items():
                for kw in keywords:
                    if kw in existing_lower:
                        counts[category] += 1
                        matched = True
                        break
                if matched: break
    return counts

def predict_gap_filling_fellowship(df, interest, pub_research, ong_research, doctor_region):
    if doctor_region and doctor_region != "Unknown":
        regional_df = df[df["Region"] == doctor_region]
        region_text = f"in {doctor_region}"
    else:
        regional_df = df
        region_text = "nationally"

    full_text = f"{str(interest)} {str(pub_research)} {str(ong_research)}".lower()
    candidates = []
    
    for fellowship, keywords in FELLOWSHIP_RULES.items():
        match_count = 0
        matched_keywords = []
        for kw in keywords:
            if kw in full_text:
                match_count += 1
                matched_keywords.append(kw)
        if match_count > 0:
            conf = min(50 + (match_count * 15), 98)
            candidates.append({"fellowship": fellowship, "score": match_count, "confidence": conf, "keywords": matched_keywords})
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    if not candidates: return None, 0, "No matching keywords found in profile."

    saturation_map = get_hospital_saturation(regional_df)
    selected = None
    skipped_log = []
    
    for cand in candidates:
        f_name = cand['fellowship']
        existing_count = saturation_map.get(f_name, 0)
        if existing_count == 0:
            selected = cand
            selected['status'] = "Strategic Gap"
            break
        else:
            skipped_log.append(f"Skipped <b>{f_name}</b> ({cand['confidence']}% match) because {existing_count} specialists already exist {region_text}.")
    
    if selected is None:
        selected = candidates[0]
        selected['status'] = "Competitive"

    reason_str = f"<b>Selected Strategy: {selected['status']} ({region_text})</b><br>Derived from: {', '.join(selected['keywords'])}."
    if skipped_log:
        reason_str += "<br><br><div style='background:#EFEBE6; padding:8px; border-radius:5px; border-left:3px solid #6D6256; font-size:0.8em; color:#5D5348;'>" + "<b>Strategic Bypass:</b><br>" + "<br>".join(skipped_log) + "</div>"
        
    return selected['fellowship'], selected['confidence'], reason_str

_PROJ_QUARTER_MONTH_START = {1: 1, 2: 4, 3: 7, 4: 10}
_PROJ_QUARTER_MONTH_END   = {1: 3, 2: 6, 3: 9, 4: 12}

def _parse_project_quarter_year(value):
    """
    Parses the cdhprojects.csv Start/End text format, e.g. 'Q3, 2026' or
    'Q3,2026' (the source data is inconsistent about the space after the
    comma - both forms appear). Returns (quarter:int, year:int) or
    (None, None) if the value doesn't match.
    """
    s = str(value).strip()
    m = re.match(r"^Q([1-4])\s*,\s*(\d{4})$", s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _parse_milestone_quarter_year(value):
    """
    Parses milestone.xlsx's actual Start Date/End Date text format, e.g.
    "Q2 '25" or "Q1 '26" - an apostrophe (not a comma) before a 2-digit
    year, with no comma at all. This is a different format from
    _parse_project_quarter_year() above (which expects 'Q3, 2026') and is
    kept fully separate so fixing this page's date parsing cannot affect
    any other page that reads quarter-year text. Returns (quarter:int,
    year:int) or (None, None) if the value doesn't match. 2-digit years
    are assumed to be 2000+yy (e.g. '25 -> 2025), which matches every
    value actually present in the source file.
    """
    s = str(value).strip()
    m = re.match(r"^Q([1-4])\s*'\s*(\d{2})$", s)
    if not m:
        return None, None
    return int(m.group(1)), 2000 + int(m.group(2))

def _project_quarter_to_start(q, year):
    return pd.Timestamp(year=year, month=_PROJ_QUARTER_MONTH_START[q], day=1)

def _project_quarter_to_end(q, year):
    return pd.Timestamp(year=year, month=_PROJ_QUARTER_MONTH_END[q], day=1) + pd.offsets.MonthEnd(0)

@st.cache_data
def load_project_data():
    """
    Loads milestone.xlsx using its REAL columns:
        #, Project Targets, Milestones, Key Activities, PIC,
        Start Date, End Date, Key Dependencies

    Column mapping used on the CDH Project Monitoring page:
      - Activity  = Milestones        (verbatim from the source file)
      - Category  = Project Targets   (verbatim - e.g. "Establishment of CDH")
      - PIC       = PIC               (verbatim)
      - Start/End = Start Date / End Date, parsed via
                    _parse_milestone_quarter_year() ("Q2 '25" style text)
      - Status    = COMPUTED from the date range, since milestone.xlsx has
                    no real status column: "Not Started" if today is before
                    Start_Date, "Completed" if today is after End_Date,
                    otherwise "In Progress".
      - Progress  = percentage of the project's Start-to-End span elapsed
                    as of today, capped at 100% and floored at 0% (same
                    calculation already used elsewhere in this app).
      - Key Dependencies is loaded too (kept for potential future display)
        but not currently rendered anywhere on the page.

    Returns an EMPTY DataFrame (not fabricated placeholder rows) if
    milestone.xlsx is missing, unreadable, or missing required columns,
    so the page can show a clear "data not found" message instead of
    silently substituting fake projects.
    """
    required = {"Project Targets", "Milestones", "PIC", "Start Date", "End Date"}
    try:
        df = pd.read_excel(f"{PROJECT_FILENAME}.xlsx")
        df.columns = [str(c).strip() for c in df.columns]
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()

        df["Activity"] = df["Milestones"].astype(str).str.strip()
        df["Category"] = df["Project Targets"].astype(str).str.strip()
        df["PIC"] = df["PIC"].astype(str).str.strip()
        if "Key Activities" in df.columns:
            df["Key Activities"] = df["Key Activities"].astype(str).str.strip()
        if "Key Dependencies" in df.columns:
            df["Key Dependencies"] = df["Key Dependencies"].astype(str).str.strip()

        start_dates, end_dates = [], []
        for _, row in df.iterrows():
            sq, sy = _parse_milestone_quarter_year(row["Start Date"])
            eq, ey = _parse_milestone_quarter_year(row["End Date"])
            start_dates.append(_project_quarter_to_start(sq, sy) if sq else pd.NaT)
            end_dates.append(_project_quarter_to_end(eq, ey) if eq else pd.NaT)
        df["Start_Date"] = start_dates
        df["End_Date"] = end_dates
        df = df.dropna(subset=["Start_Date", "End_Date"])
        if df.empty:
            df["Progress"] = pd.Series(dtype=int)
            return df
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    today = pd.Timestamp.now()

    def calc_progress(row):
        total_duration = (row['End_Date'] - row['Start_Date']).days
        elapsed = (today - row['Start_Date']).days
        if total_duration <= 0: return 0
        if elapsed < 0: return 0
        if elapsed > total_duration: return 100
        return int((elapsed / total_duration) * 100)

    def calc_status(row):
        if today < row['Start_Date']: return "Not Started"
        if today > row['End_Date']: return "Completed"
        return "In Progress"

    df['Progress'] = df.apply(calc_progress, axis=1)
    df['Status'] = df.apply(calc_status, axis=1)
    df['Duration (Weeks)'] = ((df['End_Date'] - df['Start_Date']).dt.days / 7).round(1)
    # Kept as Start/End (plain Timestamps) for compatibility with the
    # Gantt chart's x_start/x_end, mirroring the original page's columns.
    df['Start'] = df['Start_Date']
    df['End'] = df['End_Date']
    return df

# ================= UI: STYLING =================
def inject_custom_css():
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {COLOR_BG}; font-family: 'Open Sans', 'Segoe UI', sans-serif; color: {COLOR_TEXT}; }}
        [data-testid="stSidebar"] {{ background-color: {COLOR_SIDEBAR}; border-right: 1px solid #DCD5CD; }}

        /* Keep custom HTML cards within their Streamlit containers. */
        *, *::before, *::after {{ box-sizing: border-box; }}

        /* Shared authenticated workspace layout. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(#workspace-card-anchor) {{
            max-width: 1280px !important;
            margin: 0 auto !important;
            padding: 28px 32px 48px 32px !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:has(#page-content-card-anchor) {{
            background: {COLOR_CARD} !important;
            border-radius: 20px !important;
            box-shadow: 0 4px 24px rgba(93, 83, 72, 0.06) !important;
            border: 1px solid #EAE6E1 !important;
            padding: 28px 32px 32px 32px !important;
        }}

        /* Shared white content container for authenticated pages. */

        /* Global header bar. */
        .kpj-header-bar {{ padding: 4px 0 18px 0; margin-bottom: 4px; display: flex; align-items: center; gap: 20px; }}
        .kpj-logo {{ height: 60px; width: auto; }}
        .kpj-header-text-block {{ border-left: 2px solid {COLOR_PRIMARY}; padding-left: 20px; }}
        .kpj-header-title {{ color: {COLOR_TITLE}; font-size: 28px; font-weight: 800; margin: 0; line-height: 1.1; letter-spacing: -0.5px; }}
        .kpj-header-subtitle {{ color: #8C7C68; font-size: 14px; font-weight: 600; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }}
        
        .stTextInput input {{ background-color: #FFFFFF !important; color: #5D5348 !important; border: 1px solid #D0C9C0 !important; border-radius: 8px !important; padding: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }}
        div[data-testid="stForm"] {{ border: none; padding: 0; }}

        /* Shared input styling. */
        div[data-baseweb="select"] > div {{
            background-color: #FFFFFF !important;
            border: 1px solid #D0C9C0 !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important;
        }}
        div[data-baseweb="select"] > div:hover {{
            border-color: {COLOR_PRIMARY} !important;
        }}

        div[data-testid="stRadio"] label {{
            font-size: 0.88rem;
            color: {COLOR_TEXT};
            font-weight: 500;
        }}
        div[data-testid="stRadio"] [data-baseweb="radio"] div:first-child {{
            border-color: #D0C9C0 !important;
        }}

        button[data-baseweb="tab"] {{
            font-weight: 600 !important;
            color: #8C7C68 !important;
            font-size: 0.9rem !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {COLOR_TITLE} !important;
        }}
        div[data-baseweb="tab-highlight"] {{
            background-color: {COLOR_PRIMARY} !important;
        }}
        div[data-baseweb="tab-border"] {{
            background-color: #EAE6E1 !important;
        }}

        .table-container {{ background: white; border-radius: 14px; box-shadow: 0 2px 12px rgba(1, 61, 84, 0.07); border: 1px solid #EAE6E1; overflow: hidden; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        thead {{ background-color: #EAE6E1; border-bottom: 2px solid #D0C9C0; }}
        th {{ color: {COLOR_TITLE}; font-weight: 700; padding: 15px 20px; text-align: left; text-transform: uppercase; font-size: 12px; }}
        td {{ border-bottom: 1px solid #F3F0EB; padding: 15px 20px; color: {COLOR_TEXT}; vertical-align: middle; }}

        /* Specialist Directory result rows: each row is a real Streamlit
           button (not a raw <a href> anchor) so clicking a name updates
           st.query_params and reruns within the SAME session, instead of
           forcing a full browser navigation that wipes st.session_state
           (the cause of being bounced back to the login screen on every
           click). Styled to look exactly like the previous .doctor-link
           text - no button chrome - so the visual table is unchanged. */
        .specialist-row-table div[data-testid="stButton"] button {{
            background: none !important;
            border: none !important;
            box-shadow: none !important;
            padding: 15px 20px !important;
            color: {COLOR_PRIMARY} !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            text-align: left !important;
            justify-content: flex-start !important;
            border-radius: 0 !important;
            width: 100%;
        }}
        .specialist-row-table div[data-testid="stButton"] button:hover {{
            color: {COLOR_HOVER} !important;
            text-decoration: underline;
            transform: none !important;
        }}
        .specialist-row-table .specialist-row-cell {{
            padding: 15px 20px;
            border-bottom: 1px solid #F3F0EB;
            color: {COLOR_TEXT};
            font-size: 14px;
            display: flex;
            align-items: center;
            min-height: 100%;
        }}
        div[data-testid="column"]:has([id^="selected-doc-row-"]) div[data-testid="stButton"] button {{
            background-color: #F0E9DF !important;
        }}
        .specialist-row-header {{
            background-color: #EAE6E1;
            border-bottom: 2px solid #D0C9C0;
            padding: 15px 20px;
            color: {COLOR_TITLE};
            font-weight: 700;
            text-transform: uppercase;
            font-size: 12px;
        }}

        .rag-card {{ background: white; padding: 24px; border-radius: 14px; box-shadow: 0 2px 12px rgba(1, 61, 84, 0.06); margin-bottom: 10px; text-align: center; border: 1px solid #EAE6E1; border-top: 4px solid #ccc; }}
        .rag-card.red {{ border-top-color: #D32F2F; }}
        .rag-card.amber {{ border-top-color: #F2A900; }}
        .rag-card.green {{ border-top-color: #2E7D32; }}
        .rag-region {{ font-size: 0.95rem; font-weight: 700; color: {COLOR_TITLE}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
        .rag-count {{ font-size: 2.6rem; font-weight: 800; line-height: 1; }}
        .rag-count.red {{ color: #D32F2F; }}
        .rag-count.amber {{ color: #C77700; }}
        .rag-count.green {{ color: #2E7D32; }}
        .rag-status-pill {{ display: inline-block; margin-top: 8px; padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }}
        .rag-status-pill.red {{ background: #FCE4E4; color: #D32F2F; }}
        .rag-status-pill.amber {{ background: #FDF1DA; color: #C77700; }}
        .rag-status-pill.green {{ background: #E5F2E5; color: #2E7D32; }}
        
        div.stButton > button[kind="primary"] {{ background-color: {COLOR_PRIMARY} !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; box-shadow: 0 4px 10px rgba(140, 124, 104, 0.3); }}
        div.stButton > button[kind="primary"]:hover {{ background-color: {COLOR_HOVER} !important; transform: translateY(-1px); }}

        /* Secondary / default buttons (Logout, Reset, Reset Filters, etc.) -
           same border radius, font weight, and hover lift as the primary
           button above, but in the neutral executive palette so every
           button in the app reads as one consistent family regardless of
           which page it appears on. */
        div.stButton > button[kind="secondary"] {{
            background-color: {COLOR_CARD} !important;
            color: {COLOR_TITLE} !important;
            border: 1px solid #D0C9C0 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 6px rgba(93, 83, 72, 0.06);
        }}
        div.stButton > button[kind="secondary"]:hover {{
            background-color: #EAE6E1 !important;
            border-color: {COLOR_PRIMARY} !important;
            color: {COLOR_TITLE} !important;
            transform: translateY(-1px);
        }}
        div.stFormSubmitButton > button[kind="secondary"] {{
            background-color: {COLOR_CARD} !important;
            color: {COLOR_TITLE} !important;
            border: 1px solid #D0C9C0 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 6px rgba(93, 83, 72, 0.06);
        }}
        div.stFormSubmitButton > button[kind="secondary"]:hover {{
            background-color: #EAE6E1 !important;
            border-color: {COLOR_PRIMARY} !important;
            transform: translateY(-1px);
        }}

        a.doctor-link {{ color: {COLOR_PRIMARY}; font-weight: 700; text-decoration: none; }}
        
        .profile-wrapper {{ background: white; border-radius: 14px; box-shadow: 0 2px 12px rgba(1, 61, 84, 0.07); border: 1px solid #EAE6E1; overflow: hidden; margin-top: 10px; }}
        .profile-top-section {{ background: {COLOR_CARD}; padding: 24px; border-bottom: 1px solid #EAE6E1; display: flex; align-items: center; gap: 25px; }}
        .profile-photo {{ width: 100px; height: 100px; border-radius: 50%; border: 4px solid #EAE6E1; object-fit: cover; }}
        .profile-name-block h2 {{ margin: 0; font-size: 1.6rem; color: {COLOR_TITLE}; font-weight: 800; }}
        .profile-details-section {{ padding: 24px; }}
        
        .profile-row-inline {{ margin-bottom: 15px; display: flex; align-items: baseline; border-bottom: 1px dashed #EAE6E1; padding-bottom: 8px; }}
        .profile-label-text {{ font-weight: 700; color: #8C7C68; text-transform: uppercase; font-size: 0.8rem; min-width: 160px; margin-right: 10px; letter-spacing: 0.5px; }}
        .profile-value-text {{ color: {COLOR_TEXT}; font-size: 1rem; line-height: 1.5; flex: 1; font-weight: 500; }}
        
        .ai-box {{ background: {COLOR_CARD}; border: 1px solid #EAE6E1; border-left: 4px solid {COLOR_PRIMARY}; border-radius: 14px; padding: 24px; margin-top: 20px; box-shadow: 0 2px 12px rgba(1, 61, 84, 0.06); animation: fadeIn 0.6s; }}
        .ai-title {{ color: {COLOR_TITLE}; font-weight: 800; font-size: 1.2rem; margin-bottom: 10px; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}

        /* ===== STRATEGIC INSIGHTS - EXECUTIVE CARDS ===== */
        .insight-section-title {{
            color: {COLOR_TITLE};
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: 0.3px;
            margin: 4px 0 2px 0;
        }}
        .insight-section-subtitle {{
            color: #8C7C68;
            font-size: 0.82rem;
            font-weight: 500;
            margin-bottom: 16px;
            letter-spacing: 0.2px;
        }}

        .exec-card {{
            background: {COLOR_CARD};
            border-radius: 14px;
            border: 1px solid #EAE6E1;
            border-left: 4px solid {COLOR_PRIMARY};
            box-shadow: 0 2px 12px rgba(1, 61, 84, 0.06);
            padding: 24px;
            height: 280px;
            display: flex;
            flex-direction: column;
        }}
        .exec-card.risk {{ border-left-color: #8C3B3B; }}
        .exec-card.opportunity {{ border-left-color: #6E8C6B; }}
        .exec-card.action {{ border-left-color: {COLOR_PRIMARY}; }}

        .exec-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            border-bottom: 1px solid #F0ECE6;
            padding-bottom: 10px;
            margin-bottom: 14px;
        }}
        .exec-card-title {{
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: {COLOR_TITLE};
        }}
        .exec-card-count {{
            font-size: 0.72rem;
            font-weight: 700;
            color: #FFFFFF;
            padding: 2px 9px;
            border-radius: 20px;
            background: {COLOR_PRIMARY};
        }}
        .exec-card.risk .exec-card-count {{ background: #8C3B3B; }}
        .exec-card.opportunity .exec-card-count {{ background: #6E8C6B; }}

        .exec-card-body {{
            flex: 1;
            overflow-y: auto;
        }}
        .exec-item {{
            font-size: 0.86rem;
            line-height: 1.5;
            color: {COLOR_TEXT};
            padding: 9px 0;
            border-bottom: 1px dashed #F0ECE6;
        }}
        .exec-item:last-child {{ border-bottom: none; }}
        .exec-item b {{ color: {COLOR_TITLE}; font-weight: 700; }}

        .exec-item-empty {{
            font-size: 0.85rem;
            color: #A39A8E;
            font-style: italic;
            padding: 9px 0;
        }}

        .exec-action-row {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 9px 0;
            border-bottom: 1px dashed #F0ECE6;
        }}
        .exec-action-row:last-child {{ border-bottom: none; }}
        .exec-action-rank {{
            min-width: 22px;
            height: 22px;
            border-radius: 50%;
            background: {COLOR_PRIMARY};
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            margin-top: 1px;
        }}
        .exec-action-text {{
            font-size: 0.86rem;
            line-height: 1.5;
            color: {COLOR_TEXT};
        }}
        .exec-action-text b {{ color: {COLOR_TITLE}; font-weight: 700; }}

        /* Enterprise typography and spacing rhythm. */
        .page-title {{
            color: {COLOR_TITLE};
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.2px;
            margin: 0 0 16px 0;
            border-bottom: 2px solid {COLOR_PRIMARY};
            padding-bottom: 14px;
        }}
        .page-subtitle {{
            color: #8C7C68;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 0 0 20px 0;
        }}
        .section-title {{
            color: {COLOR_TITLE};
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 0.2px;
            margin: 0 0 14px 0;
        }}
        .card-title {{
            color: {COLOR_TITLE};
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.2px;
        }}
        .supporting-text {{
            color: #8C7C68;
            font-size: 0.82rem;
            font-weight: 500;
            line-height: 1.5;
        }}
        .section-divider {{
            border: none;
            border-top: 1px solid #DCD5CD;
            margin: 32px 0;
        }}


        /* ===== RISK CATEGORY BADGES (replace decorative symbols) ===== */
        .risk-badge {{
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            padding: 3px 10px;
            border-radius: 4px;
            margin-right: 8px;
            vertical-align: middle;
        }}
        .risk-badge.critical {{ background: #F4E1DC; color: #8C3B3B; border: 1px solid #D9B3AC; }}
        .risk-badge.high     {{ background: #F7E9D2; color: #97631C; border: 1px solid #E0C394; }}
        .risk-badge.moderate {{ background: #EAE6E1; color: {COLOR_TITLE}; border: 1px solid #D0C9C0; }}
        .risk-badge.covered  {{ background: #E2E9DF; color: #4A6B45; border: 1px solid #BFD2B9; }}

        /* ===== ENTERPRISE SUMMARY CARD (KPI scorecards, portfolio summaries) ===== */
        .summary-card {{
            background: #F8F6F2;
            border-radius: 8px;
            border: 1px solid #EAE6E1;
            border-left: 4px solid {COLOR_PRIMARY};
            box-shadow: none;
            padding: 18px 20px;
            height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .summary-card-label {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8C7C68;
            margin-bottom: 8px;
        }}
        .summary-card-value {{
            font-size: 2rem;
            font-weight: 800;
            color: {COLOR_TITLE};
            line-height: 1;
        }}
        .summary-card-status {{
            margin-top: 8px;
        }}

        /* ===== STRATEGIC ASSESSMENT (Geospatial structured insight) ===== */
        .assessment-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 6px;
        }}
        .assessment-row {{
            border-bottom: 1px dashed #E3DDD3;
            padding: 12px 0;
        }}
        .assessment-row:last-child {{ border-bottom: none; }}
        .assessment-label {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #8C7C68;
            margin-bottom: 5px;
        }}
        .assessment-value {{
            font-size: 0.92rem;
            color: {COLOR_TEXT};
            line-height: 1.55;
        }}
        .assessment-value b {{ color: {COLOR_TITLE}; font-weight: 700; }}

        /* ===== AUTHENTICATION / LOGIN SCREEN ===== */
        .login-logo-row {{ display: flex; justify-content: center; margin-bottom: 18px; }}
        .login-logo-row img {{ height: 56px; width: auto; }}
        .login-org-name {{
            text-align: center; color: {COLOR_TITLE}; font-size: 0.78rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;
        }}
        .login-app-title {{
            text-align: center; color: {COLOR_TITLE}; font-size: 1.6rem; font-weight: 800;
            letter-spacing: -0.3px; margin-bottom: 4px;
        }}
        .login-app-subtitle {{
            text-align: center; color: #8C7C68; font-size: 0.82rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 26px;
            padding-bottom: 18px; border-bottom: 2px solid {COLOR_PRIMARY};
        }}
        .login-field-label {{
            color: {COLOR_TITLE}; font-size: 0.78rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; margin-top: 4px;
        }}
        .login-footer-text {{
            text-align: center; color: #8C7C68; font-size: 0.78rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.8px; margin-top: 24px;
            padding-top: 18px; border-top: 1px solid #EAE6E1;
        }}
        .login-error-box {{
            background: #F4E1DC; color: #8C3B3B; border: 1px solid #D9B3AC; border-radius: 8px;
            padding: 10px 14px; font-size: 0.85rem; font-weight: 600; margin-bottom: 14px;
        }}
        .sidebar-user-box {{
            background: #FFFFFF; border: 1px solid #DCD5CD; border-radius: 10px;
            padding: 12px 14px; margin-bottom: 14px;
        }}
        .sidebar-user-name {{ color: {COLOR_TITLE}; font-weight: 700; font-size: 0.9rem; margin-bottom: 2px; }}
        .sidebar-user-role {{
            color: #8C7C68; font-size: 0.75rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.6px;
        }}

        /* ===== DIGITAL HEALTH PROJECTS ===== */
        .dhp-exec-page {{ padding-bottom: 4px; }}

        .dhp-exec-kpi-row {{ display: flex; gap: 14px; margin-bottom: 18px; }}
        .dhp-exec-kpi-card {{
            flex: 1;
            background: #FAF9F7;
            border-radius: 8px;
            border: 1px solid #E1DAD1;
            box-shadow: none;
            padding: 22px 24px;
        }}
        .dhp-exec-kpi-label {{
            font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.9px; color: #8C7C68; margin-bottom: 6px;
        }}
        .dhp-exec-kpi-value {{ font-size: 2.1rem; font-weight: 800; color: {COLOR_SECONDARY}; line-height: 1; }}
        .dhp-exec-kpi-sub {{ font-size: 0.72rem; color: #A39A8E; margin-top: 4px; font-weight: 600; }}

        .dhp-exec-summary {{
            color: {COLOR_TEXT};
            font-size: 0.88rem;
            font-weight: 500;
            margin: -4px 0 16px 0;
            line-height: 1.5;
        }}
        .dhp-exec-summary b {{ color: {COLOR_TITLE}; font-weight: 700; }}

        .dhp-exec-legend {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
        .dhp-exec-legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.7rem; color: {COLOR_TEXT}; font-weight: 600; }}
        .dhp-exec-legend-dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

        .dhp-exec-timeline-header {{
            display: flex; justify-content: space-between; align-items: flex-start;
            margin-bottom: 6px; flex-wrap: wrap; gap: 10px;
        }}

        /* The timeline card is a REAL st.container() (not manually opened/
           closed <div> tags spanning several st.markdown() calls). Streamlit
           renders each st.markdown() call as its own sibling block, so an
           opening tag in one call and a closing tag in a later call never
           actually nest the chart between them - it paints an empty box
           (this was the "empty white container" bug) while the header and
           chart float outside it as disconnected siblings. Anchoring a
           hidden marker inside the container and styling via :has() lets
           the container itself carry the card's background / radius /
           shadow / padding, with the header and chart genuinely nested
           inside a single real Streamlit block. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(#dhp-timeline-card-anchor),
        div[data-testid="stVerticalBlock"]:has(> div #dhp-timeline-card-anchor) {{
            background: transparent !important;
            border-radius: 0 !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }}
        .dhp-exec-timeline-title {{
            color: {COLOR_TITLE}; font-size: 1rem; font-weight: 700; letter-spacing: 0.1px; margin-bottom: 2px;
        }}
        .dhp-exec-timeline-sub {{ color: #8C7C68; font-size: 0.78rem; font-weight: 500; margin-bottom: 8px; }}

        /* "Generate Portfolio Insight" button - same button family as
           Search / Reset / Logout / Reset Filters (border radius, font
           weight, hover lift), sized for its position in the timeline
           header corner as a secondary action. */
        div[data-testid="column"]:has(#dhp-insight-btn-anchor) div[data-testid="stButton"] button {{
            background-color: {COLOR_CARD} !important;
            color: {COLOR_TITLE} !important;
            border: 1px solid #D0C9C0 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            box-shadow: 0 2px 6px rgba(93, 83, 72, 0.06) !important;
            transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease !important;
        }}
        div[data-testid="column"]:has(#dhp-insight-btn-anchor) div[data-testid="stButton"] button:hover {{
            background-color: #EAE6E1 !important;
            border-color: {COLOR_PRIMARY} !important;
            color: {COLOR_TITLE} !important;
            transform: translateY(-1px);
        }}

        .dhp-exec-insight-title {{ color: {COLOR_TITLE}; font-weight: 800; font-size: 1.05rem; margin-bottom: 10px; }}
        .dhp-exec-insight-body {{ color: {COLOR_TEXT}; line-height: 1.7; font-size: 0.92rem; }}
        .dhp-exec-insight-body b {{ color: {COLOR_TITLE}; }}

        /* ===== SHARED EXECUTIVE COMPONENTS ===== */

        .exec-kpi-card {{
            background: #FAF9F7;
            border-radius: 8px;
            border: 1px solid #E1DAD1;
            box-shadow: none;
            padding: 22px 24px;
            height: 100%;
        }}
        .exec-kpi-label {{
            font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.9px; color: #8C7C68; margin-bottom: 6px;
        }}
        .exec-kpi-value {{ font-size: 2.1rem; font-weight: 800; color: {COLOR_SECONDARY}; line-height: 1; }}
        .exec-kpi-sub {{ font-size: 0.72rem; color: #A39A8E; margin-top: 4px; font-weight: 600; }}

        /* Filter / search bar shell - same card treatment as the Digital
           Health Projects timeline card, used to wrap the existing search
           box and buttons on Specialist Search as one clean search section. */
        .exec-filter-bar {{
            background: {COLOR_CARD};
            border: 1px solid #EAE6E1;
            border-radius: 12px;
            padding: 14px 16px 6px 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 8px rgba(1, 61, 84, 0.04);
        }}

        /* Card anchor styles for nested Streamlit containers. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(#exec-card-anchor) {{
            background: transparent !important;
            border-radius: 0 !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }}
        /* Prevent duplicate card shells around nested filter containers. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(#exec-filter-anchor) {{
            background: transparent !important;
            border-radius: 0 !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }}
    </style>
    """, unsafe_allow_html=True)

# ================= AUTHENTICATION =================

AUTH_USER_DIRECTORY = {
    "admin":      {"password": "admin123",    "role": "Administrator"},
    "researcher": {"password": "research123", "role": "Researcher"},
    "viewer":     {"password": "viewer123",   "role": "Viewer"},
}

# Demo mode accepts any non-empty username and password.
PROTOTYPE_MODE = True

def authenticate_user(username: str, password: str):
    """
    Validates credentials.

    PROTOTYPE_MODE = True: accepts any non-empty username and password,
    assigning the "Viewer" role to whatever username was typed. No
    directory lookup happens at all - this is for demo/prototype use only.

    PROTOTYPE_MODE = False: validates against AUTH_USER_DIRECTORY.
    Returns (success: bool, role: str | None).

    --- LDAP / Active Directory migration note ---
    To swap this for LDAP or Microsoft Active Directory, replace the
    PROTOTYPE_MODE=False branch below with a directory bind/lookup, e.g.:

        from ldap3 import Server, Connection, ALL
        conn = Connection(Server("ldap://your-dc", get_info=ALL),
                           user=f"{username}@yourdomain.com", password=password)
        success = conn.bind()
        role = lookup_role_from_group_membership(conn, username)
        return success, role

    or, for Azure AD / Microsoft Entra ID via MSAL:

        result = msal_app.acquire_token_by_username_password(username, password, scopes=[...])
        success = "access_token" in result
        role = map_aad_group_to_role(result.get("id_token_claims", {}))
        return success, role

    No other part of the application needs to change - render_login_page()
    and show() only depend on this function's (success, role) contract.
    """
    if not username or not password:
        return False, None

    if PROTOTYPE_MODE:
        return True, "Viewer"

    user = AUTH_USER_DIRECTORY.get(username.strip().lower())
    if user and user["password"] == password:
        return True, user["role"]
    return False, None

def render_login_page():
    """Render the KPJ Healthcare University / CDH NEXUS Clinical OmniSuite login screen."""
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        with st.container(border=True):
            logo_html = f"<img src='data:image/png;base64,{LOGO_IMG_B64}' style='height:56px; width:auto;'>" if LOGO_IMG_B64 else ""
            error_html = "<div class='login-error-box'>Invalid username or password. Please try again.</div>" if st.session_state.get("login_error") else ""

            st.markdown(
                f"""<div class='login-logo-row'>{logo_html}</div>
                <div class='login-org-name'>KPJ Healthcare University</div>
                <div class='login-app-title'>CDH NEXUS Clinical OmniSuite</div>
                <div class='login-app-subtitle'>AI-Enabled Integrated Digital Health Portal for Academics, Researchers, & Clinicians</div>
                {error_html}""",
                unsafe_allow_html=True
            )

            with st.form("login_form", clear_on_submit=False):
                st.markdown("<div class='login-field-label'>Username</div>", unsafe_allow_html=True)
                username_input = st.text_input("Username", label_visibility="collapsed", placeholder="Enter your username")
                st.markdown("<div class='login-field-label'>Password</div>", unsafe_allow_html=True)
                password_input = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Enter your password")
                st.write("")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submitted:
                success, role = authenticate_user(username_input, password_input)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username_input.strip().lower()
                    st.session_state.role = role
                    st.session_state.login_error = False
                    st.rerun()
                else:
                    st.session_state.login_error = True
                    st.rerun()

            st.markdown("<div class='login-footer-text'>Restricted to Authorised Personnel</div>", unsafe_allow_html=True)

def render_logout_sidebar():
    """Render the signed-in user summary and logout button."""
    username = st.session_state.get("username", "")
    role = st.session_state.get("role", "")
    st.markdown(
        f"""<div class="sidebar-user-box">
            <div class="sidebar-user-name">{username if username else 'Signed in'}</div>
            <div class="sidebar-user-role">{role if role else ''}</div>
        </div>""",
        unsafe_allow_html=True
    )
    if st.button("Logout", use_container_width=True):
        for key in ["logged_in", "username", "role", "login_error"]:
            st.session_state.pop(key, None)
        st.rerun()

# ================= DATA HELPERS =================
@st.cache_data
def load_profile_image_b64():
    try:
        with open("person.png", "rb") as f: return base64.b64encode(f.read()).decode("utf-8")
    except: return ""

@st.cache_data
def load_logo_b64():
    try:
        with open("logo.png", "rb") as f: return base64.b64encode(f.read()).decode("utf-8")
    except: return ""

@st.cache_data
def load_banner_b64():
    try:
        with open(os.path.join(APP_DIR, "banner.jpeg"), "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

PROFILE_IMG_B64 = load_profile_image_b64()
LOGO_IMG_B64 = load_logo_b64()
BANNER_IMG_B64 = load_banner_b64()

def render_hero_banner():
    if not BANNER_IMG_B64:
        return
    st.markdown(
        f"""
        <div style="
            width:100%;
            border-radius:16px;
            overflow:hidden;
            box-shadow:0 6px 18px rgba(1, 61, 84, 0.11);
            margin:16px 0 24px 0;
            background:{COLOR_CARD};
        ">
            <img src="data:image/jpeg;base64,{BANNER_IMG_B64}" style="
                width:100%;
                height:auto;
                display:block;
            ">
        </div>
        """,
        unsafe_allow_html=True
    )

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda x: str(x).replace("\xa0", " ").strip())

@st.cache_data
def load_data():
    data_path = os.path.join(APP_DIR, f"{DATA_BASENAME}.csv")
    try:
        df = pd.read_csv(data_path, encoding="latin1")
    except Exception:
        st.markdown(
            f"""<div class="ai-box"><div class="ai-title">Dataset Not Found</div>
            <div style="color:{COLOR_TEXT}; line-height:1.7; font-size:0.95rem;">
            Could not find or read <b>{DATA_BASENAME}.csv</b> in the app folder.
            Expected Specialist Search dataset columns include Specialist, PublishedResearch,
            OngoingResearch, and SpecialInterest.
            </div></div>""",
            unsafe_allow_html=True
        )
        return pd.DataFrame()
    df = clean_headers(df)
    
    if "DoctorID" not in df.columns: df.insert(0, "DoctorID", [f"D{idx+1:03d}" for idx in range(len(df))])
    df["DoctorID"] = df["DoctorID"].astype(str)
    if "Region" not in df.columns: df["Region"] = "Southern"
    if "Hospital" not in df.columns: df["Hospital"] = "KPJ Specialist Hospital"
    if "Latitude" not in df.columns: df["Latitude"] = None
    if "Longitude" not in df.columns: df["Longitude"] = None

    # Coordinates are now resolved entirely from kpj_geo.csv inside
    # enrich_data_with_coords() - no inline CSV parsing duplicated here anymore.
    df = enrich_data_with_coords(df)
    return df

# ================= STRATEGIC INSIGHTS =================
def compute_strategic_insights(df):
    """
    Translates raw saturation, pipeline, and regional coverage data into
    three executive-ready buckets: Critical Risks, Opportunities, and
    Recommended Actions. Pure rule-based derivation - no new data sources,
    just a re-read of get_hospital_saturation() and the existing pending
    pipeline, broken down with regional granularity for boardroom framing.
    """
    saturation = get_hospital_saturation(df)
    regions = sorted(df["Region"].dropna().unique().tolist())

    zero_coverage = [k for k, v in saturation.items() if v == 0]
    single_point = [k for k, v in saturation.items() if v == 1]

    # Regional gap scan: for each region, which subspecialties have zero
    # coverage *in that region specifically* (not just nationally).
    region_gap_counts = {}
    for region in regions:
        region_df = df[df["Region"] == region]
        region_sat = get_hospital_saturation(region_df)
        gaps = sum(1 for v in region_sat.values() if v == 0)
        region_gap_counts[region] = gaps

    regions_with_gaps = sorted(
        [(r, g) for r, g in region_gap_counts.items() if g > 0],
        key=lambda x: x[1], reverse=True
    )

    # ---------- RISKS ----------
    # Each risk item is tagged with a governance-style category
    # (Critical / High / Moderate) rather than a decorative symbol.
    risks = []
    for sub in zero_coverage[:3]:
        risks.append({
            "category": "critical",
            "text": f"<b>{sub}</b> has no specialist coverage nationally."
        })
    for sub in single_point[:2]:
        risks.append({
            "category": "high",
            "text": f"<b>{sub}</b> is supported by a single specialist, representing a single point of failure."
        })
    if regions_with_gaps:
        worst_region, worst_gap_count = regions_with_gaps[0]
        risks.append({
            "category": "moderate",
            "text": f"<b>{worst_region} Region</b> carries {worst_gap_count} subspecialty coverage gap(s)."
        })

    # ---------- OPPORTUNITIES ----------
    pending_df = df[df["Subspecialty/Fellowship"].isna() | (df["Subspecialty/Fellowship"].astype(str).str.strip() == "")]
    pipeline_count = len(pending_df)

    interest_counts = {}
    for _, row in pending_df.iterrows():
        text = (str(row.get('SpecialInterest', '')) + " " +
                str(row.get('PublishedResearch', '')) + " " +
                str(row.get('OngoingResearch', ''))).lower()
        for cat, kws in FELLOWSHIP_RULES.items():
            if any(kw in text for kw in kws):
                interest_counts[cat] = interest_counts.get(cat, 0) + 1
                break

    top_interest = max(interest_counts.items(), key=lambda x: x[1]) if interest_counts else None

    opportunities = []
    if pipeline_count > 0:
        opportunities.append(f"<b>{pipeline_count} specialist(s)</b> in the pipeline are candidates for formal fellowship designation.")
    if top_interest:
        opportunities.append(f"The strongest emerging interest area is <b>{top_interest[0]}</b>, with {top_interest[1]} candidate(s).")
    if regions_with_gaps:
        expansion_region = regions_with_gaps[0][0]
        opportunities.append(f"<b>{expansion_region} Region</b> presents the clearest case for targeted workforce expansion.")
    if not opportunities:
        opportunities.append("No immediate pipeline opportunities identified from current data.")

    # ---------- RECOMMENDED ACTIONS (ranked) ----------
    actions = []
    if zero_coverage:
        actions.append(f"Prioritise fellowship sponsorship for <b>{zero_coverage[0]}</b> to address the most critical coverage gap.")
    if single_point:
        actions.append(f"Recruit an additional <b>{single_point[0]}</b> specialist to mitigate single-point-of-failure exposure.")
    if regions_with_gaps:
        actions.append(f"Direct recruitment and fellowship placement toward <b>{regions_with_gaps[0][0]} Region</b> to strengthen regional coverage.")
    if top_interest and len(actions) < 3:
        actions.append(f"Fast-track candidates with demonstrated interest in <b>{top_interest[0]}</b> into the formal pathway.")
    if not actions:
        actions.append("Maintain current coverage levels; no critical gaps identified at this time.")

    return {
        "risks": risks,
        "opportunities": opportunities,
        "actions": actions[:5],
        "risk_count": len(risks),
        "opportunity_count": len(opportunities),
    }


def render_strategic_insights(df):
    insights = compute_strategic_insights(df)

    st.markdown("<div class='section-title'>Strategic Insights and Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<div class='supporting-text' style='margin-bottom:16px;'>Derived from current specialist coverage, fellowship pipeline, and regional distribution data.</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if insights["risks"]:
            items_html = "".join(
                f"<div class='exec-item'><span class='risk-badge {r['category']}'>{r['category']}</span>{r['text']}</div>"
                for r in insights["risks"]
            )
        else:
            items_html = "<div class='exec-item-empty'>No critical risks identified.</div>"
        st.markdown(f"""
        <div class="exec-card risk">
            <div class="exec-card-header">
                <span class="exec-card-title">Critical Risks</span>
                <span class="exec-card-count">{insights['risk_count']}</span>
            </div>
            <div class="exec-card-body">{items_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        items_html = "".join(f"<div class='exec-item'>{o}</div>" for o in insights["opportunities"]) \
            if insights["opportunities"] else "<div class='exec-item-empty'>No opportunities identified.</div>"
        st.markdown(f"""
        <div class="exec-card opportunity">
            <div class="exec-card-header">
                <span class="exec-card-title">Opportunities</span>
                <span class="exec-card-count">{insights['opportunity_count']}</span>
            </div>
            <div class="exec-card-body">{items_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        rows_html = "".join(
            f"""<div class="exec-action-row">
                    <div class="exec-action-rank">{i+1}</div>
                    <div class="exec-action-text">{a}</div>
                </div>"""
            for i, a in enumerate(insights["actions"])
        )
        st.markdown(f"""
        <div class="exec-card action">
            <div class="exec-card-header">
                <span class="exec-card-title">Recommended Actions</span>
            </div>
            <div class="exec-card-body">{rows_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)


# ================= DASHBOARD PAGE =================
def render_dashboard(df):
    st.markdown(f"<div class='page-title'>Workforce Intelligence Overview</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>Specialist Coverage and Strategic Workforce Planning</div>", unsafe_allow_html=True)
    
    total_docs = len(df)
    sub_assigned = df[df["Subspecialty/Fellowship"].notna() & (df["Subspecialty/Fellowship"].astype(str).str.strip() != "")].shape[0]
    pending = total_docs - sub_assigned
    coverage_ratio = int((sub_assigned / total_docs) * 100) if total_docs > 0 else 0

    # Executive summary line.
    st.markdown(
        f"<div class='dhp-exec-summary'><b>{total_docs} Specialists</b> "
        f"across all regions. <b>{sub_assigned} are sub-specialized</b>, "
        f"with <b>{coverage_ratio}% coverage</b> achieved.</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Total Specialists</div>
            <div class="exec-kpi-value">{total_docs}</div>
            <div class="exec-kpi-sub">Across all regions</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Sub-Specialized</div>
            <div class="exec-kpi-value">{sub_assigned}</div>
            <div class="exec-kpi-sub">Fellowship assigned</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Pending Pathway</div>
            <div class="exec-kpi-value">{pending}</div>
            <div class="exec-kpi-sub">Awaiting designation</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Coverage Ratio</div>
            <div class="exec-kpi-value">{coverage_ratio}%</div>
            <div class="exec-kpi-sub">Specialist coverage</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ---------- Strategic Insights & Recommendations ----------
    render_strategic_insights(df)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown(f"<div class='section-title'>Current Workforce Distribution</div>", unsafe_allow_html=True)
    saturation_data = get_hospital_saturation(df)
    sat_df = pd.DataFrame(list(saturation_data.items()), columns=['Subspecialty', 'Count'])
    
    sat_df_active = sat_df[sat_df['Count'] > 0].copy()
    
    fig_tree = px.treemap(sat_df_active, path=['Subspecialty'], values='Count',
                          color='Count', color_continuous_scale=[COLOR_BG, COLOR_PRIMARY])
    fig_tree.update_layout(margin=dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_tree.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}')
    with st.container():
        st.markdown("<span id='exec-card-anchor'></span>", unsafe_allow_html=True)
        st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"<div class='section-title'>Critical Coverage Gaps</div>", unsafe_allow_html=True)
        gaps = sat_df[sat_df['Count'] == 0]['Subspecialty'].tolist()
        if gaps:
            for gap in gaps:
                st.markdown(f"<span class='risk-badge critical'>Critical</span><b>{gap}</b>", unsafe_allow_html=True)
            st.markdown("<div class='supporting-text' style='margin-top:8px;'>These subspecialties have zero specialist coverage and are prioritised for fellowship placement based on matching candidate interest.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='risk-badge covered'>Covered</span>Full spectrum coverage achieved.", unsafe_allow_html=True)

    with c_right:
        st.markdown(f"<div class='section-title'>Fellowship Pipeline Analysis</div>", unsafe_allow_html=True)
        pending_df = df[df["Subspecialty/Fellowship"].isna() | (df["Subspecialty/Fellowship"] == "")]
        if not pending_df.empty:
            interest_counts = {}
            for _, row in pending_df.iterrows():
                text = (str(row.get('SpecialInterest', '')) + " " + str(row.get('PublishedResearch', '')) + " " + str(row.get('OngoingResearch', ''))).lower()
                for cat, kws in FELLOWSHIP_RULES.items():
                    if any(kw in text for kw in kws):
                        interest_counts[cat] = interest_counts.get(cat, 0) + 1
                        break
            int_df = pd.DataFrame(list(interest_counts.items()), columns=['Potential Pathway', 'Candidates'])
            int_df = int_df.sort_values('Candidates', ascending=False)
            fig_pipe = px.pie(int_df, values='Candidates', names='Potential Pathway', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pipe.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            fig_pipe.update_traces(hovertemplate='<b>%{label}</b><br>Candidates: %{value}')
            st.plotly_chart(fig_pipe, use_container_width=True)
        else:
            st.markdown("<div class='supporting-text'>No specialists are currently pending fellowship designation.</div>", unsafe_allow_html=True)


# ================= PAGE: PROJECT MONITORING (EXECUTIVE VIEW) =================
def generate_ai_summary(df_proj, df_proj_full):
    """Generate a deterministic executive summary for the filtered project view."""
    total = len(df_proj)
    if total == 0:
        return "No activities match the current filter selection."

    completed = len(df_proj[df_proj['Status'] == 'Completed'])
    in_progress = len(df_proj[df_proj['Status'] == 'In Progress'])
    not_started = len(df_proj[df_proj['Status'] == 'Not Started'])
    avg_completion = df_proj['Progress'].mean()

    if avg_completion >= 70:
        health_phrase = "progressing well"
    elif avg_completion >= 40:
        health_phrase = "progressing, but needs attention in some areas"
    else:
        health_phrase = "at an early stage and needs close monitoring"

    cat_avg = df_proj.groupby('Category')['Progress'].mean().sort_values()
    slowest_cat = cat_avg.index[0] if not cat_avg.empty else None
    fastest_cat = cat_avg.index[-1] if not cat_avg.empty else None

    pic_counts = df_proj_full['PIC'].value_counts() if not df_proj_full.empty else pd.Series(dtype=int)
    top_pic = pic_counts.idxmax() if not pic_counts.empty else None
    top_pic_count = int(pic_counts.max()) if not pic_counts.empty else 0

    parts = [
        f"<b>Project health: {health_phrase}.</b> Across {total} tracked "
        f"activities, {completed} are completed, {in_progress} in progress, "
        f"and {not_started} not yet started (average completion: {avg_completion:.0f}%)."
    ]
    if slowest_cat and fastest_cat and slowest_cat != fastest_cat:
        parts.append(
            f"<b>{slowest_cat}</b> is the slowest-moving pillar "
            f"({cat_avg[slowest_cat]:.0f}% avg progress), while "
            f"<b>{fastest_cat}</b> is leading."
        )
    elif slowest_cat:
        parts.append(f"<b>{slowest_cat}</b> is the only active pillar currently tracked.")

    if top_pic:
        parts.append(
            f"<b>{top_pic}</b> currently holds the most activities "
            f"({top_pic_count}) - consider workload balance."
        )

    if not_started > 0:
        parts.append(f"<b>Recommendation:</b> prioritise kickoff for the {not_started} activity(ies) not yet started.")
    elif avg_completion < 40:
        parts.append("<b>Recommendation:</b> review blockers on early-stage activities to accelerate delivery.")
    else:
        parts.append("<b>Recommendation:</b> no urgent action needed - continue current pace.")

    return " ".join(parts)


def render_project_tracker():
    """Render the CDH Establishment Progress page."""
    st.markdown(f"<div class='page-title'>CDH Establishment Progress</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>CDH Programme Governance and Delivery Tracking</div>", unsafe_allow_html=True)
    executive_summary_slot = st.empty()

    df_proj_full = load_project_data()

    if df_proj_full.empty:
        st.markdown(
            f"""<div class="ai-box"><div class="ai-title">Dataset Not Found</div>
            <div style="color:{COLOR_TEXT}; line-height:1.7; font-size:0.95rem;">
            Could not find or read <b>{PROJECT_FILENAME}.xlsx</b> in the app folder, or it is
            missing one of the required columns: Project Targets, Milestones, PIC, Start Date, End Date.
            </div></div>""",
            unsafe_allow_html=True
        )
        return

    # ---------- 0. FILTER CONTROLS ----------
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        milestone_opts = ["All Milestones"] + sorted(df_proj_full['Activity'].unique().tolist())
        sel_milestone = st.selectbox("Milestone", milestone_opts, label_visibility="collapsed")
    with f2:
        pic_opts = ["All PICs"] + sorted(df_proj_full['PIC'].unique().tolist())
        sel_pic = st.selectbox("PIC", pic_opts, label_visibility="collapsed")
    with f3:
        status_opts = ["All Status"] + sorted(df_proj_full['Status'].unique().tolist())
        sel_status = st.selectbox("Status", status_opts, label_visibility="collapsed")
    with f4:
        if st.button("Reset Filters", use_container_width=True):
            st.session_state.pop("selected_activity", None)
            st.rerun()

    df_proj = df_proj_full.copy()
    if sel_milestone != "All Milestones":
        df_proj = df_proj[df_proj['Activity'] == sel_milestone]
    if sel_pic != "All PICs":
        df_proj = df_proj[df_proj['PIC'] == sel_pic]
    if sel_status != "All Status":
        df_proj = df_proj[df_proj['Status'] == sel_status]

    # ---------- 1. KPI Cards ----------
    total_acts = len(df_proj)
    active = len(df_proj[df_proj['Status'] == 'In Progress']) if total_acts else 0
    avg_prog = df_proj['Progress'].mean() if total_acts else 0
    curr_q = f"Q{(pd.Timestamp.now().month-1)//3+1} '{pd.Timestamp.now().year-2000}"

    # Executive summary line.
    executive_summary_slot.markdown(
        f"<div class='dhp-exec-summary'><b>{total_acts} Key Activities</b> "
        f"tracked for {curr_q}. <b>{active} activities</b> are currently in progress, "
        f"with <b>{avg_prog:.0f}% average completion</b>.</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Key Activities</div>
            <div class="exec-kpi-value">{total_acts}</div>
            <div class="exec-kpi-sub">Matching current filters</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Current Phase</div>
            <div class="exec-kpi-value">{curr_q}</div>
            <div class="exec-kpi-sub">Reporting quarter</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Active Tasks</div>
            <div class="exec-kpi-value">{active}</div>
            <div class="exec-kpi-sub">Currently in progress</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Avg Completion</div>
            <div class="exec-kpi-value">{avg_prog:.0f}%</div>
            <div class="exec-kpi-sub">Across filtered activities</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ---------- AI NARRATIVE SUMMARY ----------
    st.markdown(f"<div class='section-title'>AI Project Status Summary</div>", unsafe_allow_html=True)
    summary_text = generate_ai_summary(df_proj, df_proj_full)
    st.markdown(
        f"""<div class="ai-box"><div class="ai-title">Generated Summary</div>
        <div style="color:{COLOR_TEXT}; line-height:1.7; font-size:0.95rem;">{summary_text}</div></div>""",
        unsafe_allow_html=True
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ---------- 2. Gantt Chart (clickable drill-down + reliable manual select) ----------
    with st.container():
        st.markdown("<span id='exec-card-anchor'></span>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Project Master Timeline</div>", unsafe_allow_html=True)

        if not df_proj.empty:

            # Short labels for the Gantt chart only (keeps full Activity names unchanged elsewhere)
            activity_short_map = {
                "Appointment of CDH Director, Manager, 2 IT staff, 2 Academician & 2 Data Analyst& Researcher": "CDH Team Setup",
                "Appointment of CDH Director, Manager, 2 IT staff, 2 Academician & 2 Data Analyst & Researcher": "CDH Team Setup",
                "Soft setup": "Soft Setup",
                "Commence 10 CPD trainings under AI Academy": "AI Academy",
                "Initiate 3 Digital Health Projects": "3 DH Projects",
                "CDH NEXUS AI & Digital Health Summit": "NEXUS Summit",
                "CDH Digital Healthkathon": "Healthkathon",
            }

            df_proj_gantt = df_proj.copy()
            df_proj_gantt["Activity Short"] = (
                df_proj_gantt["Activity"]
                .map(activity_short_map)
                .fillna(df_proj_gantt["Activity"])
            )

            fig_gantt = px.timeline(
                df_proj_gantt,
                x_start="Start",
                x_end="End",
                y="Activity Short",
                color="Category",
                hover_name="Activity",
                hover_data={
                    "PIC": True,
                    "Status": True,
                    "Progress": True,
                    "Duration (Weeks)": True,
                    "Activity Short": False,
                },
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            today = pd.Timestamp.now()
            fig_gantt.add_vline(
                x=today,
                line_width=2,
                line_dash="dash",
                line_color="#D32F2F"
            )

            fig_gantt.add_annotation(
                x=today,
                y=1.05,
                yref="paper",
                text="Today",
                showarrow=False,
                font=dict(color="#D32F2F", size=10)
            )

            today = pd.Timestamp.now()
            fig_gantt.add_vline(x=today, line_width=2, line_dash="dash", line_color="#D32F2F")
            fig_gantt.add_annotation(x=today, y=1.05, yref="paper", text="Today", showarrow=False, font=dict(color="#D32F2F", size=10))

            fig_gantt.update_yaxes(autorange="reversed")
            fig_gantt.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=max(300, 60 + len(df_proj) * 45),
                xaxis_title="Timeline",
                yaxis_title="",
                bargap=0.3,
                xaxis=dict(showgrid=True, gridcolor='#E0E0E0', tickformat="%b '%y", dtick="M1")
            )

            gantt_event = st.plotly_chart(
                fig_gantt, use_container_width=True,
                on_select="rerun", key="gantt_chart", selection_mode="points"
            )

            if gantt_event is not None:
                sel = gantt_event.get("selection", {}) if hasattr(gantt_event, "get") else {}
                points = sel.get("points", []) if sel else []
                if points:
                    clicked_label = points[0].get("y")
                    if clicked_label:
                        st.session_state["selected_activity"] = clicked_label

            #st.markdown("<div class='supporting-text'>If clicking the chart doesn't respond on your Streamlit/Plotly version, pick the activity here instead:</div>", unsafe_allow_html=True)
            activity_options = ["-- Select an activity --"] + df_proj['Activity'].tolist()
            manual_pick = st.selectbox("Drill into activity", activity_options, key="manual_activity_picker", label_visibility="collapsed")
            if manual_pick != "-- Select an activity --":
                st.session_state["selected_activity"] = manual_pick
        else:
            st.markdown("<div class='supporting-text'>No data to display on the timeline for this filter.</div>", unsafe_allow_html=True)

    # ---------- Drill-down detail card ----------
    if st.session_state.get("selected_activity"):
        act_name = st.session_state["selected_activity"]
        match = df_proj_full[df_proj_full['Activity'] == act_name]
        if not match.empty:
            row = match.iloc[0]
            st.markdown(f"<div class='section-title' style='margin-top:24px;'>Activity Detail: {row['Activity']}</div>", unsafe_allow_html=True)
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.markdown(f"""<div class="exec-kpi-card">
                    <div class="exec-kpi-label">Category</div>
                    <div class="exec-kpi-value" style="font-size:1.1rem;">{row["Category"]}</div>
                </div>""", unsafe_allow_html=True)
            with d2:
                st.markdown(f"""<div class="exec-kpi-card">
                    <div class="exec-kpi-label">Status</div>
                    <div class="exec-kpi-value" style="font-size:1.1rem;">{row["Status"]}</div>
                </div>""", unsafe_allow_html=True)
            with d3:
                st.markdown(f"""<div class="exec-kpi-card">
                    <div class="exec-kpi-label">Completion</div>
                    <div class="exec-kpi-value">{row["Progress"]}%</div>
                </div>""", unsafe_allow_html=True)
            with d4:
                st.markdown(f"""<div class="exec-kpi-card">
                    <div class="exec-kpi-label">Duration</div>
                    <div class="exec-kpi-value">{row["Duration (Weeks)"]}w</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(
                f"<div class='supporting-text' style='margin-top:10px;'>PIC: {row['PIC']}  |  "
                f"{row['Start'].strftime('%d %b %Y')} → {row['End'].strftime('%d %b %Y')}</div>",
                unsafe_allow_html=True
            )
            if st.button("Close detail", key="close_detail"):
                st.session_state.pop("selected_activity", None)
                st.rerun()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ---------- 3. Completion Rate Chart ----------
    st.markdown(f"<div class='section-title'>Completion Rate by Activity</div>", unsafe_allow_html=True)

    if not df_proj.empty:
        chart_height = 150 + (len(df_proj) * 40)

        fig_prog = go.Figure()
        fig_prog.add_trace(go.Bar(
            y=df_proj['Activity'], x=df_proj['Progress'], orientation='h', name='Completed',
            marker=dict(color=COLOR_PRIMARY, line=dict(width=0)),
            text=df_proj['Progress'].apply(lambda x: f"{x}%"), textposition='auto', hoverinfo='y+x'
        ))
        fig_prog.add_trace(go.Bar(
            y=df_proj['Activity'], x=100 - df_proj['Progress'], orientation='h', name='Pending',
            marker=dict(color='#E0E0E0', line=dict(width=0)), textposition='none', hoverinfo='skip'
        ))
        fig_prog.update_layout(
            barmode='stack', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Completion (%)", yaxis_title="", height=chart_height,
            xaxis=dict(showgrid=False, showticklabels=True, range=[0, 100], fixedrange=True),
            yaxis=dict(showgrid=False, showline=False, fixedrange=True),
            margin=dict(l=0, r=0, t=30, b=0), showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            bargap=0.3
        )
        fig_prog.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_prog, use_container_width=True)
    else:
        st.markdown("<div class='supporting-text'>No data to display for this filter.</div>", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ---------- 4. EDITABLE Details Table ----------
    st.markdown(f"<div class='section-title'>Detailed Activity Status</div>", unsafe_allow_html=True)
    #st.markdown("<div class='supporting-text' style='margin-top:-8px; margin-bottom:14px;'>Update Progress directly in the table - Status recalculates automatically on next data refresh.</div>", unsafe_allow_html=True)

    df_display = df_proj.copy()

    edited_df = st.data_editor(
        df_display[['Activity', 'Category', 'Status', 'Progress', 'PIC']],
        use_container_width=True, hide_index=True, key="activity_editor",
        column_config={
            "Activity": st.column_config.TextColumn("Activity Name", width="large", disabled=True),
            "Category": st.column_config.TextColumn("Category", width="medium", disabled=True),
            "Status": st.column_config.TextColumn("Status", width="small", disabled=True),
            "PIC": st.column_config.TextColumn("Person In Charge", width="medium", disabled=True),
            "Progress": st.column_config.ProgressColumn("Completion", format="%d%%", min_value=0, max_value=100)
        }
    )

    if not edited_df.equals(df_display[['Activity', 'Category', 'Status', 'Progress', 'PIC']]):
        st.session_state["edited_progress"] = edited_df[['Activity', 'Progress']].to_dict('records')
        st.success("Progress updated (session only). Refresh resets to source data unless persistence is added.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ================= AI: COVERAGE GAP ANALYSIS (Geospatial) =================
@st.cache_data
def compute_region_gap_summary(df):
    """
    For each Region x Subspecialty combination, count existing specialists.
    Reuses the same saturation logic already used on the Executive Dashboard,
    but breaks it down per region so gaps can be seen spatially instead of
    only as one flat national list.
    """
    regions = sorted(df["Region"].dropna().unique().tolist())
    rows = []
    for region in regions:
        region_df = df[df["Region"] == region]
        sat = get_hospital_saturation(region_df)
        for subspecialty, count in sat.items():
            rows.append({"Region": region, "Subspecialty": subspecialty, "Count": count})
    return pd.DataFrame(rows)


def get_rag_status(count):
    """
    Converts a specialist headcount into a Red/Amber/Green status a director
    can scan instantly, instead of reading raw numbers off a 0-1 scaled axis.
      Red   = 0 specialists (critical gap, no coverage at all)
      Amber = 1 specialist  (covered but single point of failure / fragile)
      Green = 2+ specialists (adequately covered)
    """
    if count == 0:
        return "red", "No Coverage"
    elif count == 1:
        return "amber", "At Risk"
    else:
        return "green", "Covered"


def generate_strategic_assessment(gap_df, selected_sub):
    """
    Structured replacement for the former narrative-style insight box.
    Returns a dict with four governance-style fields - Current Situation,
    Risk Level, Coverage Impact, Recommended Action - instead of a single
    free-text paragraph. Underlying logic (zero/amber/covered region
    detection) is unchanged from the previous implementation.
    """
    sub_rows = gap_df[gap_df["Subspecialty"] == selected_sub]
    if sub_rows.empty:
        return {
            "situation": "No regional data is available for this subspecialty.",
            "risk_level": ("moderate", "Moderate"),
            "impact": "Unable to determine coverage impact without regional data.",
            "action": "Confirm data completeness before drawing conclusions.",
        }

    zero_regions = sub_rows[sub_rows["Count"] == 0]["Region"].tolist()
    amber_regions = sub_rows[sub_rows["Count"] == 1]["Region"].tolist()
    covered_regions = sub_rows[sub_rows["Count"] > 0].sort_values("Count", ascending=False)

    # ---- Current Situation ----
    if zero_regions:
        situation = f"No {selected_sub} specialist currently covers: {', '.join(zero_regions)}."
    else:
        situation = f"Every region has at least one {selected_sub} specialist."
    if amber_regions:
        situation += f" {', '.join(amber_regions)} {'relies' if len(amber_regions)==1 else 'rely'} on a single specialist."

    # ---- Risk Level ----
    if zero_regions:
        risk_level = ("critical", "Critical")
    elif amber_regions:
        risk_level = ("high", "High")
    else:
        risk_level = ("covered", "Low")

    # ---- Coverage Impact ----
    if not covered_regions.empty:
        top_region = covered_regions.iloc[0]
        impact = f"{top_region['Region']} holds the strongest coverage ({int(top_region['Count'])} specialist(s))."
        if zero_regions or amber_regions:
            affected = len(zero_regions) + len(amber_regions)
            impact += f" {affected} of {len(sub_rows)} region(s) are exposed to coverage risk."
    else:
        impact = f"No region currently has {selected_sub} coverage; full national exposure."

    # ---- Recommended Action ----
    if zero_regions:
        action = f"Prioritise recruitment or fellowship placement for {selected_sub} in {zero_regions[0]} first."
    elif amber_regions:
        action = f"Recruit an additional {selected_sub} specialist in {amber_regions[0]} to reduce single-point-of-failure exposure."
    else:
        action = f"Maintain current {selected_sub} coverage levels; no immediate action required."

    return {
        "situation": situation,
        "risk_level": risk_level,
        "impact": impact,
        "action": action,
    }


# ================= PAGE: GEOSPATIAL =================
def render_geospatial(df):
    st.markdown(f"<div class='page-title'>Regional Coverage Analysis</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>Specialist Distribution and Regional Risk Assessment</div>", unsafe_allow_html=True)

    # Executive summary line.
    total_specialists = len(df)
    total_regions = df["Region"].dropna().nunique()
    st.markdown(
        f"<div class='dhp-exec-summary'><b>{total_specialists} Specialists</b> "
        f"mapped across <b>{total_regions} regions</b>. Use the tabs below to view "
        f"distribution on the map or drill into regional coverage risk.</div>",
        unsafe_allow_html=True
    )

    view_tab1, view_tab2 = st.tabs(["Specialist Map", "Strategic Assessment"])

    # ---------------- TAB 1: Specialist Map ----------------
    with view_tab1:
        with st.container():
            st.markdown("<span id='exec-card-anchor'></span>", unsafe_allow_html=True)

            c1, c2 = st.columns([1, 3])
            with c1:
                filter_type = st.radio("View Mode", ["All Specialists", "By Subspecialty"], horizontal=True)
                if filter_type == "By Subspecialty":
                    selected_sub = st.selectbox("Select Subspecialty", sorted(list(FELLOWSHIP_RULES.keys())), key="geo_sub_select")
                    map_df = df[df["Subspecialty/Fellowship"].astype(str).str.contains(selected_sub.split()[0], case=False, na=False)].copy()
                else:
                    map_df = df.copy()

            map_df["Status"] = map_df["Subspecialty/Fellowship"].apply(lambda x: "Subspecialist" if pd.notna(x) and str(x).strip() != "" else "Specialist")

            if "Location_Confidence" not in map_df.columns:
                map_df["Location_Confidence"] = "Unknown"

            if map_df.empty:
                st.markdown("<div class='supporting-text'>No specialists match this filter.</div>", unsafe_allow_html=True)
            else:
                # px.scatter_map replaces the deprecated px.scatter_mapbox.
                # NOTE: scatter_map (unlike scatter_mapbox) does not support a
                # `symbol` parameter, so confidence level is instead encoded via
                # marker `size` (exact = larger, estimated/unknown = smaller) and
                # is still fully visible in the hover tooltip and the table below.
                confidence_size_map = {"Exact": 14, "Estimated (Region)": 9, "Unknown (Defaulted)": 6}
                map_df["_marker_size"] = map_df["Location_Confidence"].map(confidence_size_map).fillna(8)

                fig = px.scatter_map(
                    map_df,
                    lat="Latitude", lon="Longitude",
                    hover_name="Specialist",
                    hover_data={"Hospital": True, "Subspecialty/Fellowship": True, "Location_Confidence": True},
                    color="Status",
                    size="_marker_size",
                    size_max=14,
                    color_discrete_map={"Subspecialist": COLOR_PRIMARY, "Specialist": "#A0A0A0"},
                    zoom=5.2 if len(map_df) > 1 else 8,
                    height=500
                )
                fig.update_layout(
                    map_style="carto-positron",
                    margin=dict(t=0, l=0, r=0, b=0),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

                # The base map tile provider (CARTO/OpenStreetMap) renders a small
                # attribution credit inside the bottom-left corner of the map canvas
                # itself - this cannot be removed via Plotly layout (it's part of the
                # map tiles, required by the provider's terms of use), so instead we
                # add a clear visual gap before the next heading so nothing overlaps it.
                st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)

                estimated_count = (map_df["Location_Confidence"] != "Exact").sum()
                if estimated_count > 0:
                    st.markdown(f"<div class='supporting-text'>{estimated_count} of {len(map_df)} location(s) shown are estimated, not exact hospital coordinates.</div>", unsafe_allow_html=True)

        if not map_df.empty:
            st.markdown(f"<div class='section-title' style='margin-top:24px;'>Location Details</div>", unsafe_allow_html=True)
            st.dataframe(
                map_df[["Specialist", "Hospital", "Region", "Subspecialty/Fellowship", "Location_Confidence"]],
                hide_index=True, use_container_width=True
            )

    # ---------------- TAB 2: Strategic Assessment ----------------
    with view_tab2:
        st.markdown(
            f"<p class='supporting-text'>"
            "Select a subspecialty below. Each region is classified by coverage risk category: "
            "<b style='color:#8C3B3B;'>Critical</b> = no specialist coverage, "
            "<b style='color:#97631C;'>High</b> = single specialist (single point of failure), "
            "<b style='color:#4A6B45;'>Low</b> = adequately covered (two or more)."
            "</p>", unsafe_allow_html=True
        )

        gap_sub = st.selectbox("Subspecialty to analyse", sorted(list(FELLOWSHIP_RULES.keys())), key="gap_sub_select")
        gap_df = compute_region_gap_summary(df)

        sub_gap = gap_df[gap_df["Subspecialty"] == gap_sub].copy()
        sub_gap["rag"] = sub_gap["Count"].apply(lambda c: get_rag_status(c)[0])
        sub_gap = sub_gap.sort_values("Count", ascending=True)  # worst (red) first, left to right

        # Executive summary banner - one line, scannable in 2 seconds
        n_red = (sub_gap["rag"] == "red").sum()
        n_amber = (sub_gap["rag"] == "amber").sum()
        n_green = (sub_gap["rag"] == "green").sum()
        n_total = len(sub_gap)

        if n_red > 0:
            banner_class, banner_label = "critical", "Critical"
            banner_text = f"{n_red} of {n_total} region(s) have zero {gap_sub} coverage"
        elif n_amber > 0:
            banner_class, banner_label = "high", "High"
            banner_text = f"{n_amber} of {n_total} region(s) rely on a single {gap_sub} specialist"
        else:
            banner_class, banner_label = "covered", "Low"
            banner_text = f"All {n_total} regions have adequate {gap_sub} coverage"

        st.markdown(
            f"""<div class="summary-card" style="height:auto; flex-direction:row; align-items:center; gap:14px;">
            <span class="risk-badge {banner_class}">{banner_label} Risk</span>
            <span style="font-size:0.95rem; font-weight:600; color:{COLOR_TITLE};">{banner_text}</span>
            </div>""",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # KPI Scorecard - one tile per region, big number + RAG colour + status pill
        score_cols = st.columns(len(sub_gap)) if len(sub_gap) > 0 else []
        for col, (_, row) in zip(score_cols, sub_gap.iterrows()):
            rag_class, rag_label = get_rag_status(row["Count"])
            with col:
                st.markdown(
                    f"""<div class="rag-card {rag_class}">
                        <div class="rag-region">{row['Region']}</div>
                        <div class="rag-count {rag_class}">{int(row['Count'])}</div>
                        <div style="font-size:0.78rem; color:#8C7C68;">specialist(s)</div>
                        <div class="rag-status-pill {rag_class}">{rag_label}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        assessment = generate_strategic_assessment(gap_df, gap_sub)
        risk_class, risk_label = assessment["risk_level"]
        st.markdown(
            f"""<div class="ai-box">
            <div class="ai-title">Strategic Assessment</div>
            <div class="assessment-grid">
                <div class="assessment-row">
                    <div class="assessment-label">Current Situation</div>
                    <div class="assessment-value">{assessment['situation']}</div>
                </div>
                <div class="assessment-row">
                    <div class="assessment-label">Risk Level</div>
                    <div class="assessment-value"><span class="risk-badge {risk_class}">{risk_label}</span></div>
                </div>
                <div class="assessment-row">
                    <div class="assessment-label">Coverage Impact</div>
                    <div class="assessment-value">{assessment['impact']}</div>
                </div>
                <div class="assessment-row">
                    <div class="assessment-label">Recommended Action</div>
                    <div class="assessment-value">{assessment['action']}</div>
                </div>
            </div>
            </div>""",
            unsafe_allow_html=True
        )



# ================= DIGITAL HEALTH PROJECTS MODULE =================
# Portfolio loader and page state are scoped to the dhp_* namespace.

_DHP_QUARTER_MONTH_START = {1: 1, 2: 4, 3: 7, 4: 10}
_DHP_QUARTER_MONTH_END   = {1: 3, 2: 6, 3: 9, 4: 12}

def _dhp_parse_quarter_year(value):
    """
    Parses cdhprojects.csv Start/End text, e.g. 'Q3, 2026' or 'Q3,2026'
    (the source data is inconsistent about the space after the comma -
    both forms appear). Returns (quarter:int, year:int) or (None, None).
    """
    s = str(value).strip()
    m = re.match(r"^Q([1-4])\s*,\s*(\d{4})$", s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _dhp_quarter_to_start(q, year):
    return pd.Timestamp(year=year, month=_DHP_QUARTER_MONTH_START[q], day=1)

def _dhp_quarter_to_end(q, year):
    return pd.Timestamp(year=year, month=_DHP_QUARTER_MONTH_END[q], day=1) + pd.offsets.MonthEnd(0)

@st.cache_data
def load_digital_health_projects_data(data_mtime=None):
    """
    Loads cdhprojects.csv (ProjectName, Category, Status,
    Collaborating Organisation, Start, End).
    Independent from load_project_data() above even though both read
    quarter-year text dates, so a future change to one never silently
    affects the other. Normalises the genuine "In porgress" typo found
    in the source data to "In progress". Returns an empty DataFrame (not
    an exception) if the file is missing or malformed, so the page can
    show a clear message instead of crashing.
    """
    try:
        df = pd.read_csv(os.path.join(APP_DIR, f"{DIGITAL_HEALTH_FILENAME}.csv"))
        df.columns = [str(c).strip() for c in df.columns]
        required = {"ProjectName", "Category", "Status", "Collaborating Organisation", "Start", "End"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()

        df["ProjectName"] = df["ProjectName"].astype(str).str.strip()
        df["Category"] = df["Category"].astype(str).str.strip()
        df["Status"] = df["Status"].astype(str).str.strip().replace({"In porgress": "In Progress"})
        df["Collaborating Organisation"] = df["Collaborating Organisation"].fillna("").astype(str).str.strip()

        start_dates, end_dates = [], []
        for _, row in df.iterrows():
            sq, sy = _dhp_parse_quarter_year(row["Start"])
            eq, ey = _dhp_parse_quarter_year(row["End"])
            start_dates.append(_dhp_quarter_to_start(sq, sy) if sq else pd.NaT)
            end_dates.append(_dhp_quarter_to_end(eq, ey) if eq else pd.NaT)
        df["Start_Date"] = start_dates
        df["End_Date"] = end_dates
        df = df.dropna(subset=["Start_Date", "End_Date"])
        if df.empty:
            df["Progress"] = pd.Series(dtype=int)
            return df
    except Exception:
        return pd.DataFrame()

    today = pd.Timestamp.now()
    def calc_progress(row):
        total_duration = (row['End_Date'] - row['Start_Date']).days
        elapsed = (today - row['Start_Date']).days
        if total_duration <= 0: return 0
        if elapsed < 0: return 0
        if elapsed > total_duration: return 100
        return int((elapsed / total_duration) * 100)

    df['Progress'] = df.apply(calc_progress, axis=1)
    return df

def get_digital_health_projects_data():
    data_path = os.path.join(APP_DIR, f"{DIGITAL_HEALTH_FILENAME}.csv")
    try:
        data_mtime = os.path.getmtime(data_path)
    except OSError:
        data_mtime = None
    return load_digital_health_projects_data(data_mtime)

def apply_digital_health_project_filters(df, category, status, project_name):
    """Applies the Digital Health Projects sidebar filter selections. 'All ...' means no filter on that field."""
    filtered = df.copy()
    if category and category != "All Categories":
        filtered = filtered[filtered["Category"] == category]
    if status and status != "All Status":
        filtered = filtered[filtered["Status"] == status]
    if project_name and project_name != "All Projects":
        filtered = filtered[filtered["ProjectName"] == project_name]
    return filtered

def render_digital_health_progress_table(df_filtered):
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>CDH AI and Digital Health Project Progress</div>", unsafe_allow_html=True)

    table_df = df_filtered[["ProjectName", "Status", "Collaborating Organisation", "Progress"]].copy()
    table_df = table_df.rename(columns={
        "ProjectName": "Project Name",
        "Progress": "Progress (%)"
    })
    table_df["Progress (%)"] = table_df["Progress (%)"].astype(int)

    if _HAS_AGGRID:
        table_df = table_df.reset_index(drop=True)
        table_df.insert(0, "No.", range(1, len(table_df) + 1))

        gb = GridOptionsBuilder.from_dataframe(table_df)
        gb.configure_default_column(sortable=True, filter=True, resizable=True)
        gb.configure_grid_options(
            domLayout="normal",
            quickFilterText=st.session_state.get("dhp_progress_table_search", "")
        )
        gb.configure_column("Project Name", minWidth=360, flex=2)
        gb.configure_column("Status", minWidth=150, flex=1)
        gb.configure_column("Collaborating Organisation", minWidth=260, flex=1)
        progress_renderer = JsCode(f"""
        class ProgressCellRenderer {{
            init(params) {{
                const value = Number(params.value || 0);
                const wrap = document.createElement('div');
                wrap.style.display = 'flex';
                wrap.style.alignItems = 'center';
                wrap.style.gap = '10px';
                wrap.style.width = '100%';

                const track = document.createElement('div');
                track.style.flex = '1';
                track.style.height = '9px';
                track.style.background = '#EEEAE3';
                track.style.borderRadius = '999px';
                track.style.overflow = 'hidden';

                const bar = document.createElement('div');
                bar.style.width = Math.max(0, Math.min(100, value)) + '%';
                bar.style.height = '100%';
                bar.style.background = '{COLOR_PRIMARY}';
                bar.style.borderRadius = '999px';
                track.appendChild(bar);

                const label = document.createElement('span');
                label.innerText = value + '%';
                label.style.minWidth = '42px';
                label.style.textAlign = 'right';
                label.style.color = '{COLOR_TEXT}';
                label.style.fontWeight = '600';

                wrap.appendChild(track);
                wrap.appendChild(label);
                this.eGui = wrap;
            }}
            getGui() {{
                return this.eGui;
            }}
        }}
        """)
        gb.configure_column("Progress (%)", minWidth=190, flex=1, cellRenderer=progress_renderer)

        search = st.text_input(
            "Search project progress table",
            key="dhp_progress_table_search",
            label_visibility="collapsed",
            placeholder="Search project, status, or organisation..."
        )
        grid_options = gb.build()
        grid_options["quickFilterText"] = search

        AgGrid(
            table_df,
            gridOptions=grid_options,
            height=min(420, 95 + (len(table_df) * 42)),
            fit_columns_on_grid_load=True,
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.NO_UPDATE,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
            theme="streamlit"
        )
    else:
        search = st.text_input(
            "Search project progress table",
            key="dhp_progress_table_search_fallback",
            label_visibility="collapsed",
            placeholder="Search project, status, or organisation..."
        )
        if search:
            search_mask = table_df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            table_df = table_df[search_mask]
        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True,
            height=min(420, 95 + (len(table_df) * 42)),
            column_config={
                "Project Name": st.column_config.TextColumn("Project Name", width="large"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Collaborating Organisation": st.column_config.TextColumn("Collaborating Organisation", width="medium"),
                "Progress (%)": st.column_config.ProgressColumn("Progress (%)", format="%d%%", min_value=0, max_value=100)
            }
        )

def generate_digital_health_portfolio_insight(df_filtered):
    """Generate a deterministic portfolio summary for the filtered project set."""
    def _pluralize(n, noun):
        return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

    total = len(df_filtered)
    if total == 0:
        return "No projects match the current filter selection."

    avg_progress = df_filtered['Progress'].mean()
    active_df = df_filtered[df_filtered['Status'] != 'In Planning']
    planning_df = df_filtered[df_filtered['Status'] == 'In Planning']

    if avg_progress >= 60:
        health_phrase = "a strong delivery stage"
    elif avg_progress >= 30:
        health_phrase = "a moderate delivery stage"
    else:
        health_phrase = "an early delivery stage"

    cat_counts = df_filtered['Category'].value_counts()
    top_category = cat_counts.idxmax() if not cat_counts.empty else None
    top_category_count = int(cat_counts.max()) if not cat_counts.empty else 0

    approaching_df = df_filtered[df_filtered['Progress'] >= 50].sort_values('Progress', ascending=False)

    parts = []
    parts.append(
        f"The portfolio is at <b>{health_phrase}</b>, with an average progress of "
        f"<b>{avg_progress:.0f}%</b> across {_pluralize(total, 'project')}."
    )
    parts.append(
        f"<b>{_pluralize(len(active_df), 'project')}</b> are currently active (Alpha, Beta, or In Progress), "
        f"while <b>{_pluralize(len(planning_df), 'project')}</b> "
        f"{'remains' if len(planning_df) == 1 else 'remain'} in the planning phase."
    )
    if top_category and top_category_count > 1:
        parts.append(
            f"<b>{top_category}</b> holds the highest concentration of initiatives, "
            f"with {_pluralize(top_category_count, 'project')}."
        )
    if not approaching_df.empty:
        names = ", ".join(approaching_df['ProjectName'].head(2).tolist())
        verb = "is" if len(approaching_df) == 1 else "are"
        parts.append(
            f"<b>{_pluralize(len(approaching_df), 'project')}</b> {verb} approaching the midpoint of delivery "
            f"(up to {int(approaching_df['Progress'].max())}% complete), led by {names}."
        )
    else:
        parts.append("No projects have yet reached the midpoint of their delivery timeline.")

    if len(planning_df) > len(active_df):
        parts.append("The pipeline currently outweighs active delivery, suggesting an upcoming wave of execution as planning-phase projects transition forward.")
    elif total and len(active_df) >= total * 0.7:
        parts.append("The majority of the portfolio has moved into active delivery, indicating strong execution momentum.")

    return " ".join(parts)

def render_digital_health_sidebar_filters():
    """Render sidebar filters for the Digital Health Projects page."""
    st.markdown("---")
    st.markdown(f"<div style='color:{COLOR_TITLE}; font-weight:bold; margin-bottom:5px; text-transform:uppercase; font-size:0.78rem; letter-spacing:0.5px;'>Filter Scope</div>", unsafe_allow_html=True)

    df = get_digital_health_projects_data()
    if df.empty:
        st.markdown(f"<div class='supporting-text'>Dataset not found. Add {DIGITAL_HEALTH_FILENAME}.csv to the app folder.</div>", unsafe_allow_html=True)
        return

    categories = ["All Categories"] + sorted(df["Category"].dropna().unique().tolist())
    statuses = ["All Status"] + sorted(df["Status"].dropna().unique().tolist())
    project_names = ["All Projects"] + sorted(df["ProjectName"].dropna().unique().tolist())

    st.selectbox("Category", categories, key="dhp_category")
    st.selectbox("Status", statuses, key="dhp_status")
    st.selectbox("Project Name", project_names, key="dhp_project_name")

    if st.button("Reset Filters", use_container_width=True, key="dhp_reset_filters"):
        for k in ["dhp_category", "dhp_status", "dhp_project_name", "dhp_selected_project"]:
            st.session_state.pop(k, None)
        st.rerun()

def render_digital_health_projects():
    """Render the Digital Health Projects portfolio page."""
    st.markdown("<div class='dhp-exec-page'>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-title'>CDH AI and Digital Health Project Portfolio</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>Executive Strategic Portfolio Overview</div>", unsafe_allow_html=True)

    df = get_digital_health_projects_data()
    if df.empty:
        st.markdown(
            f"""<div class="ai-box"><div class="ai-title">Dataset Not Found</div>
            <div style="color:{COLOR_TEXT}; line-height:1.7; font-size:0.95rem;">
            Could not find or read <b>{DIGITAL_HEALTH_FILENAME}.csv</b> in the app folder.
            Expected columns: ProjectName, Category, Status, Collaborating Organisation, Start, End.
            </div></div>""",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ---- Reuse the existing sidebar filter state and the existing filter
    # function exactly as before. Nothing about this contract changes. ----
    category = st.session_state.get("dhp_category", "All Categories")
    status = st.session_state.get("dhp_status", "All Status")
    project_name_filter = st.session_state.get("dhp_project_name", "All Projects")
    df_filtered = apply_digital_health_project_filters(df, category, status, project_name_filter)

    if df_filtered.empty:
        st.markdown("<div class='supporting-text'>No projects match the current filter selection. Use Reset Filters in the sidebar to clear.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ---------- EXECUTIVE SUMMARY LINE ----------
    total_projects = len(df_filtered)
    avg_progress = df_filtered['Progress'].mean() if total_projects else 0
    active_count = len(df_filtered[df_filtered['Status'] != 'In Planning'])
    total_categories = df_filtered['Category'].nunique()
    planning_count = len(df_filtered[df_filtered['Status'] == 'In Planning'])

    st.markdown(
        f"<div class='dhp-exec-summary'><b>{total_projects} Strategic Digital Health Projects</b> "
        f"across <b>{total_categories} categories</b>. <b>{active_count} projects</b> are currently active.</div>",
        unsafe_allow_html=True
    )

    # ---------- KPI SCORECARD — four identical cards ----------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="dhp-exec-kpi-card">
            <div class="dhp-exec-kpi-label">Total Projects</div>
            <div class="dhp-exec-kpi-value">{total_projects}</div>
            <div class="dhp-exec-kpi-sub">Across {total_categories} categories</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="dhp-exec-kpi-card">
            <div class="dhp-exec-kpi-label">Average Progress</div>
            <div class="dhp-exec-kpi-value">{avg_progress:.0f}%</div>
            <div class="dhp-exec-kpi-sub">Portfolio-wide, to date</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="dhp-exec-kpi-card">
            <div class="dhp-exec-kpi-label">Active Projects</div>
            <div class="dhp-exec-kpi-value">{active_count}</div>
            <div class="dhp-exec-kpi-sub">Alpha, Beta or In Progress</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="dhp-exec-kpi-card">
            <div class="dhp-exec-kpi-label">Planning Phase</div>
            <div class="dhp-exec-kpi-value">{planning_count}</div>
            <div class="dhp-exec-kpi-sub">In the pipeline</div>
        </div>""", unsafe_allow_html=True)

    # Project delivery timeline.
    status_colors = {
        "In Planning": "#9CA3AF",     # Light Grey
        "In Discussion": "#8C7C68",   # Warm neutral
        "In Progress": "#1F6FB2",     # Blue
        "Level I Alpha": "#E08E2B",   # Orange
        "Level II Alpha": "#C77720",  # Orange variant
        "Level I Beta": "#7B5EA7",    # Purple
        "Completed": "#3F8F5B",       # Green - included for forward compatibility; no rows currently use it
    }

    # Only show legend entries for statuses actually present in the current
    # (filtered) data, so the legend never references a status with zero bars.
    legend_items = [(label, color) for label, color in status_colors.items() if label in df_filtered["Status"].unique()]
    legend_html = "".join(
        f"<span class='dhp-exec-legend-item'><span class='dhp-exec-legend-dot' style='background:{color};'></span>{label}</span>"
        for label, color in legend_items
    )

    # Anchor the timeline card to a real Streamlit container.
    with st.container():
        st.markdown("<span id='dhp-timeline-card-anchor'></span>", unsafe_allow_html=True)

        header_col, button_col = st.columns([5, 1.3])
        with header_col:
            st.markdown(
                f"""<div class='dhp-exec-timeline-title'>Project Portfolio Status</div>
                <div class='dhp-exec-timeline-sub'>Each bar spans a project's planned start to end date; the solid fill shows progress to date.</div>
                <div class='dhp-exec-legend'>{legend_html}</div>""",
                unsafe_allow_html=True
            )
        with button_col:
            st.markdown("<span id='dhp-insight-btn-anchor'></span>", unsafe_allow_html=True)
            st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
            generate_clicked = st.button("✨ Generate Portfolio Insight", key="dhp_generate_insight", use_container_width=True)

        timeline_df = df_filtered.sort_values('Start_Date', ascending=True).copy()
        timeline_df['Duration_Days'] = (timeline_df['End_Date'] - timeline_df['Start_Date']).dt.days
        timeline_df['Progress_Date'] = timeline_df['Start_Date'] + pd.to_timedelta(
            timeline_df['Duration_Days'] * timeline_df['Progress'] / 100, unit='D'
        )
        progress_days = (timeline_df['Progress_Date'] - timeline_df['Start_Date']).dt.days.clip(lower=1)

        def _truncate_name(name, n=36):
            return name if len(name) <= n else name[:n - 1] + "…"
        timeline_df['Label'] = timeline_df['ProjectName'].apply(_truncate_name)

        bar_colors = [status_colors.get(s, "#9CA3AF") for s in timeline_df['Status']]

        # Selected project naturally renders as the only row.

        fig = go.Figure()
        # Faded track: full planned Start -> End span
        fig.add_trace(go.Bar(
            y=timeline_df['Label'], x=timeline_df['Duration_Days'], base=timeline_df['Start_Date'],
            orientation='h',
            marker=dict(color=bar_colors, opacity=0.22, line=dict(width=0)),
            hoverinfo='skip', showlegend=False,
        ))
        # Solid fill: Start -> progress-to-date
        fig.add_trace(go.Bar(
            y=timeline_df['Label'], x=progress_days, base=timeline_df['Start_Date'],
            orientation='h',
            marker=dict(color=bar_colors, line=dict(width=0)),
            customdata=timeline_df[['ProjectName', 'Category', 'Status', 'Collaborating Organisation', 'Progress', 'Start_Date', 'End_Date']],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Category: %{customdata[1]}<br>"
                "Status: %{customdata[2]}<br>"
                "Collaborating Organisation: %{customdata[3]}<br>"
                "Progress: %{customdata[4]}%<br>"
                "Start: %{customdata[5]|%b %Y}<br>"
                "End: %{customdata[6]|%b %Y}<extra></extra>"
            ),
            showlegend=False,
        ))
        fig.update_layout(
            barmode='overlay',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=max(380, min(580, 44 + len(timeline_df) * 36)),
            margin=dict(t=10, b=6, l=4, r=10),
            xaxis=dict(showgrid=True, gridcolor='#EEEAE3', tickfont=dict(size=11, color=COLOR_TEXT)),
            yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=12, color=COLOR_TEXT)),
            hoverlabel=dict(bgcolor="white", font_size=12, font_color=COLOR_TEXT, bordercolor="#EAE6E1"),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Generate portfolio insight on demand.
        current_filter_signature = (category, status, project_name_filter)
        if st.session_state.get("dhp_insight_filter_signature") != current_filter_signature:
            st.session_state.pop("dhp_insight_text", None)
            st.session_state["dhp_insight_filter_signature"] = current_filter_signature

        if generate_clicked:
            st.session_state["dhp_insight_text"] = generate_digital_health_portfolio_insight(df_filtered)

        if st.session_state.get("dhp_insight_text"):
            with st.expander("Portfolio Insight", expanded=True):
                st.markdown(
                    f"""<div class="dhp-exec-insight-title">Executive Summary</div>
                    <div class="dhp-exec-insight-body">{st.session_state['dhp_insight_text']}</div>""",
                    unsafe_allow_html=True
                )

    render_digital_health_progress_table(df_filtered)

    st.markdown("</div>", unsafe_allow_html=True)
# ================= SEARCH PAGE =================
def render_search(df):
    st.markdown(f"<div class='page-title'>Specialist Search</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-subtitle'>Career and Research Pathway Lookup</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='dhp-exec-summary'><b>{len(df)} Specialist Profiles</b> available for keyword search across research, clinical interests, and specialist names.</div>",
        unsafe_allow_html=True
    )

    if "q" not in st.session_state:
        st.session_state["q"] = st.query_params.get("q", "")
    elif "q" in st.query_params and st.query_params.get("q", "") != st.session_state["q"] and "doctor" in st.query_params:
        st.session_state["q"] = st.query_params.get("q", "")

    with st.container():
        st.markdown("<span id='exec-filter-anchor'></span>", unsafe_allow_html=True)
        col_form, col_reset = st.columns([5, 1])
        with col_form:
            with st.form("search", clear_on_submit=False):
                c_in, c_btn = st.columns([4, 1])
                with c_in: q_input = st.text_input("Search", value=st.session_state["q"], placeholder="Search research titles, clinical interests, or specialist names...", label_visibility="collapsed")
                with c_btn: is_search = st.form_submit_button("Search", type="primary", use_container_width=True)
        with col_reset:
            if st.button("Reset", type="secondary", use_container_width=True):
                st.session_state["q"] = ""
                st.query_params.clear()
                st.rerun()

    if is_search: st.session_state["q"] = q_input; st.rerun()

    q = st.session_state["q"]
    filtered = pd.DataFrame()

    if q.strip():
        df_norm = df.copy()
        df_norm["_text"] = df_norm[["Specialist","PublishedResearch","OngoingResearch","SpecialInterest"]].fillna("").agg(" ".join, axis=1).str.lower()
        filtered = df[df_norm["_text"].str.contains(q.lower(), na=False)]
        st.markdown(f"<div class='supporting-text' style='margin-bottom:10px; font-size:0.9rem;'>{len(filtered)} specialist(s) match the current search criteria.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='supporting-text'>Enter keywords to search across published research, ongoing studies, clinical interests, and specialist names.</div>", unsafe_allow_html=True)

    selected_doc_id = st.query_params.get("doctor", None)

    if selected_doc_id:
        c_list, c_prof = st.columns([6, 4], gap="large")
    else:
        c_list, c_prof = st.columns([1, 0.01])

    with c_list:
        if not filtered.empty:
            st.markdown('<div class="table-container specialist-row-table">', unsafe_allow_html=True)

            header_cols = st.columns([3, 2, 5])
            with header_cols[0]: st.markdown("<div class='specialist-row-header'>Name</div>", unsafe_allow_html=True)
            with header_cols[1]: st.markdown("<div class='specialist-row-header'>Subspecialty</div>", unsafe_allow_html=True)
            with header_cols[2]: st.markdown("<div class='specialist-row-header'>Published Research</div>", unsafe_allow_html=True)

            for _, row in filtered.iterrows():
                doc_id = str(row["DoctorID"])
                is_selected = doc_id == str(selected_doc_id)

                sub = row.get("Subspecialty/Fellowship", "")
                badge = (
                    "<span class='risk-badge moderate'>Pathway Pending</span>"
                    if pd.isna(sub) or str(sub).strip() == ""
                    else str(sub)
                )

                raw_res = row.get("PublishedResearch", "")
                if pd.isna(raw_res) or str(raw_res).strip() == "":
                    res = "No published research listed"
                else:
                    res = str(raw_res).strip()

                row_cols = st.columns([3, 2, 5])
                with row_cols[0]:
                    if is_selected:
                        st.markdown(f"<span id='selected-doc-row-{doc_id}'></span>", unsafe_allow_html=True)
                    if st.button(row["Specialist"], key=f"doc_row_{doc_id}", use_container_width=True):
                        st.query_params["doctor"] = doc_id
                        if q: st.query_params["q"] = q
                        st.rerun()
                with row_cols[1]:
                    st.markdown(f"<div class='specialist-row-cell {'selected' if is_selected else ''}'>{badge}</div>", unsafe_allow_html=True)
                with row_cols[2]:
                    st.markdown(f"<div class='specialist-row-cell {'selected' if is_selected else ''}' style='color:#666;'>{res}</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with c_prof:
        if selected_doc_id:
            record = df[df["DoctorID"]==selected_doc_id].iloc[0].to_dict()
            img = f'<img src="data:image/png;base64,{PROFILE_IMG_B64}" class="profile-photo">' if PROFILE_IMG_B64 else ""
            
            card = f"""<div class="profile-wrapper"><div class="profile-top-section">{img}<div class="profile-name-block"><h2>{record.get('Specialist')}</h2><p>{record.get('Hospital','KPJ')}</p><p>{record.get('Region','Malaysia')}</p></div></div><div class="profile-details-section">"""
            for k,v in [("Specialty", "Specialty"), ("Subspecialty", "Subspecialty/Fellowship"), ("Clinical Interest", "SpecialInterest"), ("Ongoing Research", "OngoingResearch"), ("Published Research", "PublishedResearch")]:
                val = record.get(v, "")
                if pd.notna(val) and str(val).strip(): card += f'<div class="profile-row-inline"><span class="profile-label-text">{k}:</span><span class="profile-value-text">{val}</span></div>'
            st.markdown(card + "</div></div>", unsafe_allow_html=True)

            sub = record.get("Subspecialty/Fellowship", "")
            if pd.isna(sub) or str(sub).strip() == "":
                st.write("")
                if st.button("Fellowship Pathway Recommendation", type="primary", use_container_width=True):
                    with st.spinner("Generating assessment..."):
                        import time; time.sleep(0.5)
                        s, c, r = predict_gap_filling_fellowship(df, record.get("SpecialInterest",""), record.get("PublishedResearch",""), record.get("OngoingResearch",""), record.get("Region"))
                    if s: st.markdown(f"""<div class="ai-box"><div class="ai-title">Fellowship Pathway Recommendation</div><div style="font-size:1.2rem; font-weight:600; color:#5D5348;">{s}</div><div style="margin-top:5px; font-size:0.9em;">{r}</div><div style="margin-top:15px; font-weight:bold; color:#8C7C68;">Confidence: {c}%</div></div>""", unsafe_allow_html=True)
                    else: st.markdown("<div class='supporting-text'><span class='risk-badge moderate'>Moderate</span>Insufficient profile data to generate a recommendation.</div>", unsafe_allow_html=True)

@st.cache_data
def load_research_programmes_data(data_mtime=None):
    try:
        df = pd.read_csv(os.path.join(APP_DIR, "cpd.csv"), encoding="latin1")
        df.columns = [str(c).strip() for c in df.columns]

        date_col = "JANGKAAN TARIKH" if "JANGKAAN TARIKH" in df.columns else "Date"
        required = {"Category", "CourseName", date_col, "Status"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()

        out = pd.DataFrame()
        out["Category"] = df["Category"].fillna("").astype(str).str.strip()
        out["Programme Name"] = df["CourseName"].fillna("").astype(str).str.strip()
        out["Expected Date"] = df[date_col].fillna("").astype(str).str.strip()
        out["Status"] = df["Status"].fillna("").astype(str).str.strip().str.title()
        return out
    except Exception:
        return pd.DataFrame()

def get_research_programmes_data():
    data_path = os.path.join(APP_DIR, "cpd.csv")
    try:
        data_mtime = os.path.getmtime(data_path)
    except OSError:
        data_mtime = None
    return load_research_programmes_data(data_mtime)

def render_research_kpi_scorecards(df):
    total_programmes = len(df)
    status_counts = df["Status"].value_counts()
    achieved = int(status_counts.get("Achieved", 0))
    in_progress = int(status_counts.get("In Progress", 0))
    planned = int(status_counts.get("Planned", 0))
   ## not_achieved = int(status_counts.get("Not Achieved", 0))
    achieved_rate = (achieved / total_programmes * 100) if total_programmes else 0

    st.markdown(f"<div class='section-title'>KPJHS Research KPI Summary</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Total Programmes</div>
            <div class="exec-kpi-value">{total_programmes}</div>
            <div class="exec-kpi-sub">Across all categories</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Achieved</div>
            <div class="exec-kpi-value">{achieved}</div>
            <div class="exec-kpi-sub">{achieved_rate:.0f}% completion rate</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">In Progress</div>
            <div class="exec-kpi-value">{in_progress}</div>
            <div class="exec-kpi-sub">Currently active</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="exec-kpi-card">
            <div class="exec-kpi-label">Planned</div>
            <div class="exec-kpi-value">{planned}</div>
            <div class="exec-kpi-sub">Upcoming programmes</div>
        </div>""", unsafe_allow_html=True)
  ##  with k5:
    ##    st.markdown(f"""<div class="exec-kpi-card">
          ##  <div class="exec-kpi-label">Not Achieved</div>
      ##      <div class="exec-kpi-value">{not_achieved}</div>
        ##    <div class="exec-kpi-sub">Requires attention</div>
        ##</div>""", unsafe_allow_html=True)

def render_research_programmes_table(df):
    st.markdown(f"<div class='section-title'>KPJHS AI and Digital Health Research Programme List</div>", unsafe_allow_html=True)

    def _research_table_height(row_count):
        return min(620, 48 + (max(row_count, 1) * 41))

    if _HAS_AGGRID:
        search = st.text_input(
            "Search research programme list",
            key="research_programme_search",
            label_visibility="collapsed",
            placeholder="Search programme, category, date, or status..."
        )
        table_df = df.copy()
        if search:
            search_mask = table_df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            table_df = table_df[search_mask]

        table_df = table_df.reset_index(drop=True)
        table_df.insert(0, "No.", range(1, len(table_df) + 1))

        gb = GridOptionsBuilder.from_dataframe(table_df)
        gb.configure_default_column(sortable=True, filter=True, resizable=True)
        gb.configure_grid_options(
            domLayout="normal",
            quickFilterText=""
        )
        gb.configure_column("No.", minWidth=70, maxWidth=80, flex=0)
        gb.configure_column("Category", minWidth=140, flex=1)
        gb.configure_column("Programme Name", minWidth=420, flex=2)
        gb.configure_column("Expected Date", minWidth=150, flex=1)
        status_renderer = JsCode("""
        function(params) {
            const raw = String(params.value || '');
            const key = raw.toLowerCase();
            const colors = {
                'achieved': { bg: '#E7F4EA', fg: '#2F6B3A', bd: '#B9DEC1' },
                'in progress': { bg: '#FFF4CC', fg: '#8A6500', bd: '#E7CC72' },
                'planned': { bg: '#E7F0FA', fg: '#275F91', bd: '#B8D3EB' },
              ##  'not achieved': { bg: '#FCE8E6', fg: '#9A3B30', bd: '#E7B8B0' }
            };
            const c = colors[key] || { bg: '#EEEAE3', fg: '#5D5348', bd: '#D8D0C5' };
            const badge = document.createElement('span');
            badge.innerText = raw;
            badge.style.display = 'inline-flex';
            badge.style.alignItems = 'center';
            badge.style.padding = '4px 10px';
            badge.style.borderRadius = '999px';
            badge.style.background = c.bg;
            badge.style.color = c.fg;
            badge.style.border = '1px solid ' + c.bd;
            badge.style.fontWeight = '700';
            badge.style.fontSize = '0.78rem';
            badge.style.lineHeight = '1.2';
            return badge;
        }
        """)
        gb.configure_column("Status", minWidth=150, flex=1, cellRenderer=status_renderer)

        grid_options = gb.build()

        AgGrid(
            table_df,
            gridOptions=grid_options,
            height=_research_table_height(len(table_df)),
            fit_columns_on_grid_load=True,
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.NO_UPDATE,
            data_return_mode=DataReturnMode.FILTERED,
            allow_unsafe_jscode=True,
            theme="streamlit"
        )
    else:
        search = st.text_input(
            "Search research programme list",
            key="research_programme_search_fallback",
            label_visibility="collapsed",
            placeholder="Search programme, category, date, or status..."
        )
        table_df = df.copy()
        if search:
            search_mask = table_df.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            table_df = table_df[search_mask]

        table_df = table_df.reset_index(drop=True)
        table_df.insert(0, "No.", range(1, len(table_df) + 1))

        def _research_status_badge_style(value):
            colors = {
                "Achieved": ("#E7F4EA", "#2F6B3A", "#B9DEC1"),
                "In Progress": ("#FFF4CC", "#8A6500", "#E7CC72"),
                "Planned": ("#E7F0FA", "#275F91", "#B8D3EB"),
              ##  "Not Achieved": ("#FCE8E6", "#9A3B30", "#E7B8B0"),
            }
            bg, fg, border = colors.get(str(value), ("#EEEAE3", COLOR_TEXT, "#D8D0C5"))
            return (
                f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
                "border-radius: 999px; font-weight: 700; text-align: center;"
            )

        styled_table = table_df.style.map(_research_status_badge_style, subset=["Status"])
        st.dataframe(styled_table, use_container_width=True, hide_index=True, height=_research_table_height(len(table_df)))

def render_research_placeholder():
    st.markdown(f"<div class='page-title'>CDH AI Academy AI & Digital Health Training Programmes 2026</div>", unsafe_allow_html=True)
    df = get_research_programmes_data()
    if df.empty:
        st.markdown(
            f"""<div class="ai-box"><div class="ai-title">Dataset Not Found</div>
            <div style="color:{COLOR_TEXT}; line-height:1.7; font-size:0.95rem;">
            Could not find or read <b>cpd.csv</b> in the app folder.
            Expected columns: Category, CourseName, JANGKAAN TARIKH or Date, Status.
            </div></div>""",
            unsafe_allow_html=True
        )
        return
    render_research_kpi_scorecards(df)
    render_research_programmes_table(df)

# ================= MAIN =================
def show():
    inject_custom_css()

    # Authentication gate.
    if not st.session_state.get("logged_in", False):
        render_login_page()
        return

    # Backward-compatible authenticated marker.
    st.markdown("<div class='app-authenticated'></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<h3 style='color:{COLOR_TITLE}'>Navigation</h3>", unsafe_allow_html=True)
        page = st.radio("Go to", [
            # "Workforce Intelligence Overview",  # Temporarily disabled.
            # "Regional Coverage Analysis",      # Temporarily disabled.
            "CDH AI and Digital Health Project Portfolio",
            "CDH Establishment Progress",
            "CDH AI Academy AI & Digital Health Training Porgrammes 2026",
            "Specialist Search"
        ], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("<div style='font-size:0.8em; color:#8C7C68;'>CDH NEXUS Clinical OmniSuite v2.6<br>Restricted to Authorised Personnel</div>", unsafe_allow_html=True)
        
        # Temporarily disabled with Workforce Intelligence Overview navigation.
        # if page == "Workforce Intelligence Overview":
        #     st.markdown("---")
        #     st.markdown(f"<div style='color:{COLOR_TITLE}; font-weight:bold; margin-bottom:5px; text-transform:uppercase; font-size:0.78rem; letter-spacing:0.5px;'>Filter Scope</div>", unsafe_allow_html=True)
        #     df = load_data()
        #     if not df.empty:
        #         all_regions = sorted(df["Region"].dropna().unique().tolist())
        #         sel_region = st.selectbox("Region", ["All Regions"] + all_regions)
        #         if sel_region != "All Regions": st.session_state['filter_region'] = sel_region
        #         else: st.session_state['filter_region'] = None

        if page == "CDH AI and Digital Health Project Portfolio":
            render_digital_health_sidebar_filters()

        # Logout control.
        st.markdown("---")
        render_logout_sidebar()

    # Global application header.
    logo_header_html = f'<img src="data:image/png;base64,{LOGO_IMG_B64}" class="kpj-logo">' if LOGO_IMG_B64 else ""
    st.markdown(f"""<div class="kpj-header-bar">{logo_header_html}<div class="kpj-header-text-block"><div class="kpj-header-title">CDH NEXUS Clinical OmniSuite</div><div class="kpj-header-subtitle">AI-Enabled Integrated Digital Health Portal for Academics, Researchers, & Clinicians</div></div></div>""", unsafe_allow_html=True)

    # Shared content container.
    with st.container():
        st.markdown("<span id='page-content-card-anchor'></span>", unsafe_allow_html=True)
        render_hero_banner()

        # Route pages that do not require the specialist dataset first.
        if page == "CDH AI and Digital Health Project Portfolio":
            render_digital_health_projects()
            return

        if page == "CDH AI Academy AI & Digital Health Training Porgrammes 2026":
            render_research_placeholder()
            return

        if 'df' not in locals(): df = load_data()
        if df.empty: return

        # Temporarily disabled with Workforce Intelligence Overview navigation.
        # if page == "Workforce Intelligence Overview" and 'filter_region' in st.session_state and st.session_state['filter_region']:
        #     df = df[df["Region"] == st.session_state['filter_region']]

        if page == "Specialist Search": render_search(df)
        # elif page == "Workforce Intelligence Overview": render_dashboard(df)  # Temporarily disabled.
        # elif page == "Regional Coverage Analysis": render_geospatial(df)      # Temporarily disabled.
        elif page == "CDH Establishment Progress": render_project_tracker()

if __name__ == "__main__":
    show()


