"""
BuildWise AI — Intelligent Construction & Home Planning Assistant (single-file Streamlit app)

Run:   streamlit run app.py --theme.base light        (or simply:  python app.py)

How this file is organised (search for the ===== headers):
  1. Settings (.env)        6. PDF reports
  2. Data                   7. Saved projects
  3. Theme & styling        8. Pages (welcome, dashboard, estimator, approvals, consultant, redesign)
  4. Cost calculations      9. App shell (sidebar, navigation, theme toggle)
  5. AI (OpenAI, RAG, images)  10. Embedded images

Security: the OpenAI API key is typed on the welcome screen and kept only in st.session_state for the
current browser session. It is never hard-coded, written to disk, logged, or included in any report.
"""
import base64
import copy
import io
import json
import os
import sys
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as pgo
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from fpdf import FPDF
from openai import (APIConnectionError, APIStatusError, AuthenticationError, BadRequestError, NotFoundError, OpenAI,
                    PermissionDeniedError, RateLimitError)
from PIL import Image


# ====================================================================================================
# 1. SETTINGS (.env)
# ====================================================================================================
load_dotenv()  # reads optional settings from .env (never the API key)
warnings.filterwarnings("ignore", message=".*langchain-community.*")  # informational notice from LangChain


def _env_list(name: str, default: str) -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


CHAT_MODELS = _env_list("CHAT_MODELS", "gpt-4.1-mini,gpt-4o-mini,gpt-5-mini,gpt-4.1,gpt-4o")  # first available is used
IMAGE_MODELS = _env_list("IMAGE_MODELS", "gpt-image-1,gpt-image-1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
PROJECTS_FILE = Path(__file__).resolve().parent / os.getenv("PROJECTS_FILE", "buildwise_projects.json")  # relative to app.py


# ====================================================================================================
# 2. DATA — sample rate card (edit before relying on totals) and general approval guidance
# ====================================================================================================
SAMPLE_RATES = {'rate_date': '2026-10-30',
 'status': 'SAMPLE',
 'note': 'Sample rates for demonstration only. They are NOT live market prices - replace them with current local '
         'quotes before relying on any total.',
 'materials': {'cement': {'label': 'Cement', 'unit': 'bag (50 kg)', 'rate': 1450, 'per_sqft': 0.45},
               'steel': {'label': 'Steel (rebar)', 'unit': 'kg', 'rate': 260, 'per_sqft': 3.6},
               'bricks': {'label': 'Bricks', 'unit': 'brick', 'rate': 17, 'per_sqft': 22},
               'sand': {'label': 'Sand', 'unit': 'cft', 'rate': 80, 'per_sqft': 1.8},
               'crush': {'label': 'Crush (aggregate)', 'unit': 'cft', 'rate': 170, 'per_sqft': 0.9}},
 'grey_labour_per_sqft': 450,
 'finishing_per_sqft': {'Basic': 1800, 'Standard': 3000, 'Premium': 5000},
 'finishing_split': {'Flooring & tiles': 0.26,
                     'Woodwork & doors': 0.18,
                     'Plumbing & sanitary': 0.16,
                     'Electrical': 0.14,
                     'Kitchen & fixtures': 0.14,
                     'Paint & plaster': 0.12},
 'construction_factor': {'RCC frame with brick infill': 1.0, 'Load-bearing brick masonry': 0.92},
 'contingency_pct': 5,
 'marla_sizes': {'225 sq ft': 225, '250 sq ft': 250, '272.25 sq ft': 272.25},
 'plot_sizes': {'5 Marla': 5, '10 Marla': 10, '1 Kanal': 20},
 'cities': ['Lahore', 'Islamabad', 'Rawalpindi', 'Karachi', 'Faisalabad', 'Multan', 'Peshawar', 'Other']}

APPROVALS = {'disclaimer': 'General guidance compiled for planning purposes. It is not an official list. Requirements vary by '
               'authority, scheme, plot and project and change over time - always confirm with the authority.',
 'authorities': {'LDA': {'name': 'Lahore Development Authority',
                         'area': 'Lahore',
                         'website': 'https://www.lda.gop.pk'},
                 'CDA': {'name': 'Capital Development Authority',
                         'area': 'Islamabad',
                         'website': 'https://www.cda.gov.pk'},
                 'DHA': {'name': 'Defence Housing Authority', 'area': 'Lahore · Karachi · Islamabad', 'website': ''},
                 'Society': {'name': 'Private housing society', 'area': 'Society bylaws apply', 'website': ''}},
 'checklists': {'New house': [['Proof of ownership',
                               'Registry / transfer letter / allotment letter, fard where applicable'],
                              ['Owner identity documents', 'CNIC copies of all owners'],
                              ['Site plan / location plan', 'Plot dimensions, access road and orientation'],
                              ['Architectural drawings',
                               'Plans, elevations and sections signed by a registered architect'],
                              ['Structural drawings & calculations', 'Signed by a licensed structural engineer'],
                              ['Utility / other NOCs where required',
                               'Depends on scheme and project - ask the authority'],
                              ['Application form & fee challan', 'Format and fee as issued by the authority'],
                              ['Undertakings / affidavits', 'As required by the authority']],
                'Extension / addition': [['Proof of ownership', 'Registry / transfer letter / allotment letter'],
                                         ['Previously approved building plan',
                                          'Copy of the existing approval and completion documents'],
                                         ['Drawings of proposed changes',
                                          'Existing and proposed plans clearly marked'],
                                         ['Structural assessment',
                                          "Engineer's confirmation that the structure can take the addition"],
                                         ['Application form & fee challan',
                                          'Format and fee as issued by the authority'],
                                         ['Undertakings / affidavits', 'As required by the authority']],
                'Commercial': [['Proof of ownership & land use',
                                'Ownership documents and confirmation of permitted commercial use'],
                               ['Architectural drawings', 'Signed by a registered architect'],
                               ['Structural drawings & calculations', 'Signed by a licensed structural engineer'],
                               ['Fire safety & services drawings', 'Where required for the building category'],
                               ['Parking & traffic arrangements', 'Where required by the authority'],
                               ['Environmental / other NOCs', 'Depends on size and use - ask the authority'],
                               ['Application form & fee challan', 'Format and fee as issued by the authority']]},
 'procedure': [['Prepare documents', 'Collect ownership papers and professionally signed drawings'],
               ['Submit application', 'File with the fee challan at the authority or its online portal'],
               ['Scrutiny & site inspection', 'The authority reviews drawings and may inspect the site'],
               ['Address observations', 'Revise drawings if objections are raised'],
               ['Approval issued', 'Start construction only after written approval']]}


# ====================================================================================================
# 3. THEME & STYLING — one design system for every page; LIGHT/DARK only swap colour tokens
# ====================================================================================================
CSS = """
/* ==========================================================================
   BuildWise AI — one shared styling system for every page.
   Colours come from CSS variables injected by the LIGHT / DARK dictionaries below (light / dark),
   so this file never changes between themes.
   ========================================================================== */

/* ---------- base ---------- */
html, body, .stApp, .stApp :is(p, li, label, input, textarea, button, td, th, div, a, small, b, strong) { font-family: 'DM Sans', system-ui, sans-serif; }
.stApp { background: var(--bg); color: var(--ink); transition: background-color .45s ease, color .45s ease; }
.stApp p, .stApp li, .stApp label, .stApp span { color: inherit; }
h1, h2, h3, h4, .serif, .serif *, .big, .quote, .num, .kp, .qrow .v { font-family: 'Cormorant Garamond', Georgia, serif !important; letter-spacing: .1px; }
.stApp [data-testid="stMarkdownContainer"] { color: var(--ink); }
h1, h2, h3 { color: var(--ink) !important; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 1.4rem 2rem 3rem; max-width: 1440px; }
a { color: var(--green); }
hr { border-color: var(--line); }

/* ---------- motion ---------- */
@keyframes bwRise  { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes bwSpin  { to { transform: rotate(360deg); } }
@keyframes bwSweep { 0% { transform: translateX(-120%); } 100% { transform: translateX(320%); } }
@keyframes bwFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes bwZoom  { from { transform: scale(1.06); } to { transform: scale(1); } }
[class*="st-key-page_"] { animation: bwRise .55s cubic-bezier(.2,.7,.2,1) both; }  /* page transition */
.rise { animation: bwRise .7s cubic-bezier(.2,.7,.2,1) both; }
.d1 { animation-delay: .06s; } .d2 { animation-delay: .12s; } .d3 { animation-delay: .18s; } .d4 { animation-delay: .24s; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--line); transition: background-color .45s; }
[data-testid="stSidebarHeader"] { min-height: 0; height: 2.6rem; padding: .6rem .8rem 0; }
[data-testid="stSidebarUserContent"] { padding-top: .2rem; }
[data-testid="stSidebar"] .stButton button > div, [data-testid="stSidebar"] .stButton button > div > span { justify-content: flex-start; width: 100%; gap: 10px; }
.brand { display: flex; align-items: center; gap: 12px; padding: 0 4px 10px; }
.brand b { display: block; font-family: 'Cormorant Garamond', serif; font-size: 27px; font-weight: 700; line-height: 1; color: var(--ink); }
.brand small { display: block; font-size: 8.5px; letter-spacing: 1.6px; color: var(--mut); margin-top: 5px; white-space: nowrap; }
[data-testid="stSidebar"] .stButton button { justify-content: flex-start; border: 0; background: transparent; color: var(--ink2); font-size: 14.5px; padding: 10px 14px; border-radius: 12px; box-shadow: none; transition: background .25s, padding .25s, color .25s; }
[data-testid="stSidebar"] .stButton button:hover { background: var(--beige); padding-left: 18px; color: var(--ink); transform: none; }
[data-testid="stSidebar"] .stButton button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { background: var(--card) !important; color: var(--ink) !important; font-weight: 600; box-shadow: var(--shadow); border: 0 !important; border-left: 3px solid var(--green) !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p { color: var(--ink) !important; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:after { display: none; }
[data-testid="stSidebar"] .stButton button p { font-size: 14.5px; }
.promo { border-radius: 18px; overflow: hidden; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); margin: 14px 0 10px; }
.promo img { display: block; width: 100%; height: 130px; object-fit: cover; transition: transform .8s; }
.promo:hover img { transform: scale(1.05); }
.promo > div { padding: 14px 16px 16px; }
.status { display: flex; align-items: center; gap: 10px; font-size: 12.5px; color: var(--mut); padding: 4px 2px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; box-shadow: 0 0 0 4px var(--gsoft); }

/* ---------- cards (keyed containers + html) ---------- */
.card, div[class*="st-key-card"], div[class*="st-key-feat"], div[class*="st-key-auth"], div[class*="st-key-sty"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow);
  transition: transform .35s cubic-bezier(.2,.7,.2,1), box-shadow .35s, background-color .45s, border-color .45s;
}
div[class*="st-key-card"] { padding: 22px 22px 18px; }
div[class*="st-key-card"]:hover, .card.lift:hover { transform: translateY(-3px); box-shadow: var(--shadow2); }
.card { padding: 22px; }
/* cards that hold inputs stay still on hover (no lift while typing) */
.st-key-card_connect:hover, .st-key-card_inputs:hover, .st-key-card_chat:hover, .st-key-card_check:hover,
.st-key-card_studio:hover, .st-key-card_canvas:hover, .st-key-card_explain:hover, .st-key-card_kb:hover, .st-key-card_bylaws:hover { transform: none; box-shadow: var(--shadow); }
div[class*="st-key-sty"] { border-radius: 18px; }
/* markdown blocks inside click-cards keep their natural height */
div[class*="st-key-sty"] [data-testid="stMarkdownContainer"], div[class*="st-key-auth"] [data-testid="stMarkdownContainer"], div[class*="st-key-feat"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
div[class*="st-key-sty"] .stMarkdown, div[class*="st-key-auth"] .stMarkdown { margin-bottom: 0 !important; }
.hd { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 6px; }
.hd h3 { margin: 0; padding: 0 !important; font-size: 22px; font-weight: 600; display: flex; align-items: center; gap: 9px; line-height: 1.15; }
.hd h3 svg { flex: none; }
.hd h3 svg { color: var(--terra); }
.mut, .note { color: var(--mut) !important; } .note { font-size: 12px; line-height: 1.5; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px; border-radius: 999px; font-size: 11.5px; font-weight: 600; letter-spacing: .3px; white-space: nowrap; }
.pill.g { background: var(--gsoft); color: var(--green); } .pill.t { background: var(--tsoft); color: var(--tdeep); }
.pill.n { background: var(--card2); color: var(--ink2); border: 1px solid var(--line); }

/* invisible full-card click target: a keyed st.button stretched over its card */
div[class*="st-key-feat"], div[class*="st-key-auth"], div[class*="st-key-sty"] { position: relative; overflow: hidden; }
div[class*="st-key-hit_"] { position: absolute !important; inset: 0; z-index: 5; margin: 0 !important; }
div[class*="st-key-hit_"] button { width: 100%; height: 100%; opacity: 0; cursor: pointer; }
div[class*="st-key-auth"]:hover, div[class*="st-key-sty"]:hover { transform: translateY(-4px); box-shadow: var(--shadow2); }

/* ---------- page header banner ---------- */
.banner { position: relative; overflow: hidden; display: flex; flex-wrap: wrap; min-height: 160px; border-radius: 22px; background: linear-gradient(120deg, var(--card) 45%, var(--card2)); border: 1px solid var(--line); box-shadow: var(--shadow); margin-bottom: 18px; animation: bwRise .6s both; }
.banner .txt { flex: 1 1 420px; padding: 24px 28px; position: relative; z-index: 1; }
.banner .crumb { font-size: 11.5px; letter-spacing: 1.8px; font-weight: 600; color: var(--mut); }
.banner h1 { font-size: 42px !important; font-weight: 600; margin: 6px 0 4px; line-height: 1.04; padding: 0; }
.banner p { margin: 0; color: var(--ink2); font-size: 15px; max-width: 560px; }
.banner .img { flex: 0 1 440px; min-height: 160px; position: relative; }
.banner .img img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; -webkit-mask-image: linear-gradient(to right, transparent, #000 40%); mask-image: linear-gradient(to right, transparent, #000 40%); opacity: var(--imgop); }

/* ---------- dashboard ---------- */
.hero { position: relative; border-radius: 24px; overflow: hidden; min-height: 250px; padding: 26px 4px 10px; animation: bwRise .6s both; }
.hero img { position: absolute; right: 0; top: 0; height: 100%; width: 62%; object-fit: cover; -webkit-mask-image: linear-gradient(to right, transparent 10%, #000 60%); mask-image: linear-gradient(to right, transparent 10%, #000 60%); opacity: var(--imgop); animation: bwZoom 1.6s cubic-bezier(.2,.7,.2,1) both; border-radius: 24px; }
.hero h1 { font-size: 52px !important; font-weight: 600; margin: 0; line-height: 1.04; position: relative; padding: 0; }
.hero p { position: relative; font-size: 17px; color: var(--ink2); margin: 8px 0 0; }
.quote { position: absolute; right: 22px; bottom: 22px; padding: 11px 18px; border-radius: 16px; background: var(--glass); backdrop-filter: blur(12px); box-shadow: var(--shadow); font: italic 600 20px 'Cormorant Garamond', serif; color: var(--ink); }
div[class*="st-key-feat"] { min-height: 236px; padding: 20px; border: 0; }
div[class*="st-key-feat"]:hover { transform: translateY(-6px); box-shadow: var(--shadow2); }
div[class*="st-key-feat"]:hover .fimg { transform: scale(1.06); }
div[class*="st-key-feat"]:hover .go { transform: translateX(4px); }
.st-key-feat_cost { background: linear-gradient(150deg, #3C5C48, #2A4334) !important; }
.st-key-feat_appr { background: linear-gradient(160deg, var(--f2a), var(--f2b)) !important; border: 1px solid var(--line) !important; }
.st-key-feat_chat { background: linear-gradient(160deg, var(--f3a), var(--f3b)) !important; }
.st-key-feat_room { background: linear-gradient(160deg, var(--card), var(--card2)) !important; border: 1px solid var(--line) !important; }
.feat { position: relative; z-index: 1; }
.feat .row { display: flex; justify-content: space-between; align-items: center; }
.feat .fi { width: 46px; height: 46px; border-radius: 13px; display: flex; align-items: center; justify-content: center; }
.feat .go { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: transform .3s; font-size: 15px; }
.feat h4 { font: 600 24px/1.08 'Cormorant Garamond', serif; margin: 14px 0 0; max-width: 230px; }
.feat p { font-size: 13px; line-height: 1.5; margin: 8px 0 0; max-width: 64%; opacity: .9; }
.hstats { position: relative; display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
.hstats span { padding: 9px 14px; border-radius: 13px; background: var(--glass); border: 1px solid var(--line); font-size: 13px; color: var(--ink2); backdrop-filter: blur(8px); }
.hstats b { color: var(--ink); }
.fimg { position: absolute; right: -20px; bottom: -16px; height: 44%; width: 62%; object-fit: cover; object-position: right bottom; z-index: 0; transition: transform .6s cubic-bezier(.2,.7,.2,1); -webkit-mask-image: radial-gradient(ellipse 100% 100% at 100% 100%, #000 50%, transparent 100%); mask-image: radial-gradient(ellipse 100% 100% at 100% 100%, #000 50%, transparent 100%); }
.big { font: 700 46px/1 'Cormorant Garamond', serif; color: var(--ink); }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; padding: 12px 0; margin: 14px 0 4px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); font-size: 13.5px; }
.stats b { display: block; font-size: 15px; color: var(--ink); }
.mrow { display: flex; align-items: center; gap: 12px; padding: 7px 6px; border-radius: 12px; transition: background .2s; }
.mrow:hover { background: var(--card2); }
.swatch { width: 46px; height: 38px; border-radius: 9px; flex: none; box-shadow: inset 0 0 0 1px rgba(0,0,0,.06); }
.track { height: 6px; border-radius: 6px; background: var(--line); overflow: hidden; margin-top: 7px; }
.track i { display: block; height: 100%; border-radius: 6px; background: var(--terra); animation: fillIn 1.1s cubic-bezier(.2,.7,.2,1) both; }
@keyframes fillIn { from { width: 0; } }
.step { position: relative; display: flex; gap: 13px; padding: 6px 0 10px; }
.step .b { z-index: 1; width: 24px; height: 24px; border-radius: 50%; flex: none; display: flex; align-items: center; justify-content: center; border: 1.6px solid var(--green); font-size: 12px; color: var(--on); }
.steps { position: relative; } .steps:before { content: ""; position: absolute; left: 11.5px; top: 12px; bottom: 18px; width: 1.5px; background: var(--line2); }
.step .t { font-size: 13.5px; font-weight: 500; color: var(--ink); } .step .s { font-size: 12px; color: var(--mut); }
.photo { border-radius: 14px; overflow: hidden; } .photo img { display: block; width: 100%; height: 160px; object-fit: cover; transition: transform .8s; } .photo:hover img { transform: scale(1.05); }
.bubble { padding: 13px 15px; border-radius: 4px 16px 16px 16px; background: var(--card2); border: 1px solid var(--line); font-size: 13.5px; line-height: 1.55; color: var(--ink); }

/* ---------- estimator ---------- */
.total { position: relative; overflow: hidden; border-radius: 20px; padding: 24px 28px; color: #F4EFE6; background: linear-gradient(140deg, #3C5C48, #243A2D); box-shadow: var(--shadow2); animation: bwRise .6s both; }
.total:before { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px); background-size: 36px 36px; }
.total * { position: relative; color: #F4EFE6; }
.total .num { font: 700 54px/.95 'Cormorant Garamond', serif; }
.total .kp { font: 700 28px/1.1 'Cormorant Garamond', serif; }
.total .bar { display: flex; height: 10px; border-radius: 10px; overflow: hidden; margin-top: 18px; background: rgba(255,255,255,.12); }
.qrow { display: flex; align-items: center; gap: 12px; padding: 8px 6px; border-radius: 12px; transition: background .2s; }
.qrow:hover { background: var(--card2); }
.qrow .v { font: 700 22px 'Cormorant Garamond', serif; color: var(--ink); }
.tip { display: flex; gap: 10px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 12px; font-size: 13px; line-height: 1.5; margin-bottom: 8px; transition: background .2s; }
.tip:hover { background: var(--card2); }

/* ---------- welcome ---------- */
.wl { position: relative; overflow: hidden; border-radius: 26px; min-height: calc(100vh - 3.2rem); padding: 36px 44px; color: #F4EFE6; background: var(--wlbg); display: flex; flex-direction: column; gap: 26px; transition: background .45s; }
.wl:before { content: ""; position: absolute; inset: 0; background-image: linear-gradient(rgba(255,255,255,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.045) 1px, transparent 1px); background-size: 44px 44px; }
.wl:after { content: ""; position: absolute; right: -160px; top: -160px; width: 520px; height: 520px; border-radius: 50%; background: radial-gradient(circle, rgba(217,160,127,.28), transparent 65%); }
.wl > * { position: relative; z-index: 1; }
.wl h2 { color: #F4EFE6 !important; font-size: 46px !important; font-weight: 600; line-height: 1.03; margin: 0; padding: 0; }
.wl .lead { color: rgba(244,239,230,.78); font-size: 15.5px; line-height: 1.6; max-width: 470px; margin: 12px 0 0; }
.collage { position: relative; flex: 1; min-height: 360px; }
.collage .a { position: absolute; left: 0; right: 40px; top: 0; height: 270px; border-radius: 20px; overflow: hidden; border: 6px solid rgba(244,239,230,.1); box-shadow: 0 30px 60px -20px rgba(0,0,0,.55); }
.collage .b { position: absolute; right: 0; top: 230px; width: 54%; height: 140px; border-radius: 18px; overflow: hidden; border: 5px solid rgba(244,239,230,.14); box-shadow: 0 24px 48px -16px rgba(0,0,0,.55); }
.collage img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .8s; }
.collage div:hover img { transform: scale(1.05); }
.float { position: absolute; padding: 14px 16px; border-radius: 16px; background: rgba(252,249,243,.95); color: #23241F; box-shadow: 0 20px 40px -14px rgba(0,0,0,.45); animation: bwFloat 6s ease-in-out infinite; }
.float * { color: #23241F; }
.feats { display: flex; flex-wrap: wrap; gap: 22px; font-size: 13px; color: rgba(244,239,230,.8); }
.feats span:before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #D9A07F; margin-right: 8px; }
.stepper { display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--mut); margin-top: 6px; }
.stepper i { flex: 1; height: 1px; background: var(--line2); }
.stepper b { width: 22px; height: 22px; border-radius: 50%; border: 1.5px solid var(--line2); display: inline-flex; align-items: center; justify-content: center; font-size: 11px; margin-right: 6px; }
.stepper .on { color: var(--green); font-weight: 600; } .stepper .on b { background: var(--green); border-color: var(--green); color: var(--on); }
.wtitle { font-size: 54px !important; line-height: 1 !important; font-weight: 600; margin: 30px 0 12px !important; padding: 0 !important; }
.wtitle em { color: var(--terra); }
.trio { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; font-size: 12.5px; line-height: 1.45; color: var(--mut); margin-top: 6px; }
.trio b { display: block; color: var(--ink); margin-bottom: 3px; }
.loader { padding: 14px; border-radius: 14px; background: var(--card2); border: 1px solid var(--line); }
.loader .row { display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--ink2); }
.spin { width: 16px; height: 16px; border-radius: 50%; border: 2px solid var(--line2); border-top-color: var(--green); animation: bwSpin .8s linear infinite; flex: none; }
.sweep { height: 5px; border-radius: 5px; background: var(--line); overflow: hidden; margin-top: 10px; }
.sweep i { display: block; width: 35%; height: 100%; background: var(--terra); border-radius: 5px; animation: bwSweep 1.2s ease-in-out infinite; }
.msg { display: flex; gap: 10px; padding: 13px 14px; border-radius: 14px; font-size: 13.5px; line-height: 1.45; margin-top: 10px; }
.msg.err { background: var(--tsoft); color: var(--tdeep); } .msg.ok { background: var(--gsoft); color: var(--green); font-weight: 600; }

/* ---------- native widgets themed ---------- */
.stApp label, [data-testid="stWidgetLabel"] p { color: var(--ink2) !important; font-weight: 600; font-size: 12.5px; }
.stTextInput input, .stNumberInput input, .stTextArea textarea, [data-baseweb="select"] > div, [data-testid="stChatInput"] textarea {
  background: var(--card) !important; color: var(--ink) !important; border-color: var(--line2) !important; border-radius: 12px !important; font-size: 14.5px; }
.stTextInput [data-baseweb="input"], .stNumberInput [data-baseweb="input"], .stTextArea [data-baseweb="textarea"] { background: var(--card) !important; border: 1.5px solid var(--line2) !important; border-radius: 13px !important; transition: border-color .2s, box-shadow .2s; }
.stTextInput [data-baseweb="input"]:focus-within, .stNumberInput [data-baseweb="input"]:focus-within, .stTextArea [data-baseweb="textarea"]:focus-within { border-color: var(--green) !important; box-shadow: 0 0 0 4px var(--gsoft); }
.stTextInput input::placeholder, .stTextArea textarea::placeholder, [data-testid="stChatInput"] textarea::placeholder { color: var(--mut) !important; opacity: 1; }
[data-baseweb="select"] > div { border: 1.5px solid var(--line2) !important; } [data-baseweb="select"] svg { color: var(--ink2); }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] { background: var(--card) !important; }
[data-baseweb="popover"] li { color: var(--ink) !important; } [data-baseweb="popover"] li:hover, [data-baseweb="popover"] [aria-selected="true"] { background: var(--card2) !important; }
.stNumberInput button { background: var(--card2) !important; color: var(--ink) !important; border: 0 !important; }
/* buttons */
.stButton button, .stDownloadButton button, .stFormSubmitButton button, .stLinkButton a {
  border-radius: 13px; min-height: 44px; font-weight: 600; border: 1.5px solid var(--line2); background: var(--card); color: var(--ink);
  transition: transform .2s, box-shadow .25s, background .25s, border-color .25s; position: relative; overflow: hidden; }
.stButton button:hover, .stDownloadButton button:hover, .stFormSubmitButton button:hover, .stLinkButton a:hover { transform: translateY(-2px); border-color: var(--green); color: var(--green); background: var(--card2); }
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"], .stFormSubmitButton button[kind="primary"],
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] { background: var(--green) !important; color: var(--on) !important; border-color: var(--green) !important; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover { box-shadow: 0 12px 24px -10px var(--green); background: var(--green2) !important; color: var(--on) !important; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p { color: var(--on) !important; }
[data-testid="stBaseButton-primary"]:after, [data-testid="stBaseButton-primaryFormSubmit"]:after { content: ""; position: absolute; top: 0; bottom: 0; width: 40%; background: linear-gradient(100deg, transparent, rgba(255,255,255,.22), transparent); transform: translateX(-120%); }
[data-testid="stBaseButton-primary"]:hover:after, [data-testid="stBaseButton-primaryFormSubmit"]:hover:after { animation: bwSweep .9s ease; }
div.st-key-generate .stButton button { background: var(--terra) !important; border-color: var(--terra) !important; color: #fff !important; }
div.st-key-generate .stButton button p { color: #fff !important; }
button:disabled { opacity: .5 !important; transform: none !important; }
/* segmented controls / pills */
[data-testid="stButtonGroup"] { gap: 5px; }
[data-testid="stButtonGroup"] button { border-radius: 10px !important; border: 1px solid var(--line) !important; background: var(--card2) !important; color: var(--ink2) !important; min-height: 38px; transition: all .25s; }
[data-testid="stButtonGroup"] button[kind*="Active"], [data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] [data-testid*="Active"] { background: var(--green) !important; color: var(--on) !important; border-color: var(--green) !important; box-shadow: 0 6px 14px -6px var(--green); }
[data-testid="stButtonGroup"] [data-testid*="Active"] p, [data-testid="stButtonGroup"] button[aria-checked="true"] p { color: var(--on) !important; }
/* checkboxes & toggles */
.stCheckbox label p, .stToggle label p { color: var(--ink) !important; font-weight: 500; font-size: 14px; }
.stCheckbox { padding: 8px 10px; border-radius: 12px; transition: background .2s; } .stCheckbox:hover { background: var(--card2); }
/* expander, uploader, alerts, chat, tables */
[data-testid="stExpander"] details { background: var(--card); border: 1px dashed var(--line2); border-radius: 16px; }
[data-testid="stExpander"] summary p { color: var(--ink) !important; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] { background: var(--card2); border: 1.5px dashed var(--line2); border-radius: 16px; transition: border-color .2s, background .2s; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--terra); }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink2) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--card) !important; }
[data-testid="stFileUploader"] [data-testid*="File"]:not([data-testid="stFileUploaderDropzone"]) { background: transparent !important; }
[data-testid="stFileUploader"] [data-testid*="File"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderFile"] { background: var(--card) !important; border-radius: 10px; }
[data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stAlert"] p { color: var(--ink) !important; }
[data-testid="stChatMessage"] { background: transparent; padding: 6px 0; animation: bwRise .45s both; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { background: var(--card2); border: 1px solid var(--line); padding: 12px 16px; border-radius: 4px 18px 18px 18px; color: var(--ink); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] { background: var(--green); border-color: var(--green); border-radius: 18px 18px 4px 18px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] * { color: var(--on) !important; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--ink); }
[data-testid="stChatInput"] { border-radius: 16px; border: 1.5px solid var(--line2) !important; background: var(--card) !important; }
[data-testid="stChatInput"] > div { background: var(--card) !important; }
[data-testid="stChatInputSubmitButton"] { background: var(--green) !important; color: var(--on) !important; border-radius: 10px !important; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border-radius: 12px; overflow: hidden; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--mut) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line) !important; border-radius: 16px; }
.stSpinner > div { color: var(--ink2); }
.stPlotlyChart { animation: bwRise .7s both; }

/* ---------- responsive ---------- */
@media (max-width: 1200px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 1rem 1.2rem 2rem; }
  .hero h1 { font-size: 42px !important; } .banner h1 { font-size: 34px !important; }
  .hero img { width: 55%; } .total .num { font-size: 44px; }
}
@media (max-width: 1100px) {
  .collage .b, .collage .float { display: none; } .collage { min-height: 300px; } .collage .a { right: 0; }
  .hstats span:nth-child(n+3) { display: none; }
}
@media (max-width: 900px) {
  .banner .img, .hero img { display: none; } .quote { display: none; }
  .wl { min-height: auto; padding: 26px; } .wl h2 { font-size: 34px !important; } .collage { min-height: 330px; }
  .wtitle { font-size: 42px !important; } .trio { grid-template-columns: 1fr; }
  .stats { grid-template-columns: 1fr 1fr; }
}
"""

LIGHT = dict(bg="#F5F0E7", side="#EEE7DA", card="#FCF9F3", card2="#F4EDE2", line="#E5DCCB", line2="#D7CBB7",
             ink="#23241F", ink2="#45443E", mut="#6F6B62", green="#2F4A3A", green2="#3E5D4A", gsoft="#E2EAE0",
             terra="#C98466", tsoft="#F4E2D6", tdeep="#93472A", beige="#EADFCF", sand="#DCCFBC", on="#F8F4EC",
             glass="rgba(252,249,243,.84)", imgop="1", wlbg="#2F4A3A",
             f2a="#F6EFE4", f2b="#ECE1D1", f3a="#DFA98B", f3b="#CF9070", f3fg="#2E1D14",
             shadow="0 1px 2px rgba(70,55,30,.05),0 10px 30px -12px rgba(70,55,30,.16)",
             shadow2="0 22px 44px -18px rgba(47,74,58,.35)")
# Dark theme: warm charcoal, muted forest green, cream text (higher contrast for readability).
DARK = dict(bg="#141512", side="#191A17", card="#1F211D", card2="#272922", line="#33362E", line2="#43463D",
            ink="#F4ECDF", ink2="#DED5C4", mut="#B2AA99", green="#7FA48A", green2="#93B79D", gsoft="#26342B",
            terra="#D29466", tsoft="#3B2C21", tdeep="#EBB48A", beige="#2C2D27", sand="#55564C", on="#10120F",
            glass="rgba(31,33,29,.86)", imgop=".62", wlbg="#121A15",
            f2a="#2A2B25", f2b="#22231E", f3a="#5A3828", f3b="#45291D", f3fg="#F6E2D2",
            shadow="0 1px 2px rgba(0,0,0,.3),0 14px 34px -14px rgba(0,0,0,.6)",
            shadow2="0 24px 48px -18px rgba(0,0,0,.75)")

FONTS = ("@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;"
         "1,500;1,600&family=DM+Sans:wght@400;500;600&display=swap');")

ICONS = {
    "calc": "M6 3h12v18H6zM9 7h6M9 11h1M12 11h1M9 14h1M12 14h1M9 17h1M12 17h1M15 14v3",
    "doc": "M7 3h7l4 4v14H7zM14 3v4h4M10 12h5M10 16h5",
    "chat": "M4 5h16v11H9l-5 4zM8 9h8M8 12h5",
    "sofa": "M5 11V8a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v3M3 12a2 2 0 0 1 4 0v2h10v-2a2 2 0 0 1 4 0v5H3zM5 17v2M19 17v2",
    "folder": "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
    "bars": "M3 21h18M5 21V10l4-3v14M9 21V5l6-3v19M15 21V9l4 2v10",
    "spark": "M12 3l1.8 4.6L18 9l-4.2 1.4L12 15l-1.8-4.6L6 9l4.2-1.4z",
    "pin": "M12 21s7-6.2 7-12a7 7 0 1 0-14 0c0 5.8 7 12 7 12Z",
}


# CSS "material swatches" used next to quantities (pure CSS, no images needed).
TEX = {
    "cement": "radial-gradient(#8f897d 1px,transparent 1.6px) 0 0/6px 6px,linear-gradient(160deg,#D8D1C5,#BDB5A8)",
    "steel": "repeating-linear-gradient(120deg,#6f726d 0 3px,#b3b6b0 3px 6px,#8a8d87 6px 9px)",
    "bricks": "linear-gradient(#E6D8C6 2px,transparent 2px) 0 0/100% 10px,linear-gradient(90deg,#E6D8C6 2px,transparent 2px) 0 0/16px 20px,#B4603F",
    "sand": "radial-gradient(#B8955F 1.4px,transparent 1.8px) 0 0/5px 5px,#D9BF92",
    "crush": "radial-gradient(#7d7a72 2px,transparent 2.6px) 0 0/7px 7px,radial-gradient(#b9b4aa 1.6px,transparent 2px) 3px 3px/7px 7px,#9d988e",
}


def dark() -> bool:
    return st.session_state.get("dark", False)


def pal() -> dict:
    return DARK if dark() else LIGHT


def apply(sidebar: bool = True):
    """Inject fonts, theme variables and the shared stylesheet."""
    vars_ = ";".join(f"--{k}:{v}" for k, v in pal().items())
    hide = "" if sidebar else ('[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],'
                               '[data-testid="collapsedControl"]{display:none!important}')
    st.html(f"<style>{FONTS}:root{{{vars_}}}{CSS}{hide}</style>")


def html(s: str):
    """Render an HTML snippet (lines are stripped so Markdown never treats them as code)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


def img(name: str) -> str:
    """Return an embedded image (see IMAGES at the end of this file) as a data URI."""
    return "data:image/webp;base64," + IMAGES[name]


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="{ICONS[name]}"/></svg>')


def pkr(n: float) -> str:
    return f"{n:,.0f}"


def short(n: float) -> str:
    return f"{n / 1e6:.1f}M" if abs(n) >= 1e6 else f"{n / 1e3:.0f}K"


def card_head(title: str, icon_name: str = "", right: str = ""):
    ic = icon(icon_name) if icon_name else ""
    html(f'<div class="hd"><h3>{ic}{title}</h3>{right}</div>')


def banner(crumb: str, title: str, subtitle: str, image: str):
    html(f"""<div class="banner"><div class="txt"><div class="crumb">TOOLS &nbsp;/&nbsp; {crumb}</div>
    <h1>{title}</h1><p>{subtitle}</p></div><div class="img"><img src="{img(image)}" alt=""></div></div>""")


def loader(text: str):
    """Branded loading indicator (render into an st.empty placeholder)."""
    return (f'<div class="loader"><div class="row"><span class="spin"></span>{text}</div>'
            f'<div class="sweep"><i></i></div></div>')


def style_fig(fig, height: int = 300):
    """Make a Plotly figure match the current theme."""
    p = pal()
    fig.update_layout(height=height, margin=dict(l=12, r=12, t=10, b=12), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans", color=p["ink2"], size=12),
                      hoverlabel=dict(bgcolor=p["ink"], font_color=p["bg"], bordercolor=p["ink"]),
                      legend=dict(orientation="h", y=-0.12, x=0), transition_duration=500)
    fig.update_xaxes(gridcolor=p["line"], zeroline=False, linecolor=p["line2"], automargin=True)
    fig.update_yaxes(gridcolor=p["line"], zeroline=False, linecolor=p["line2"], automargin=True, tickformat="~s")
    return fig


# ====================================================================================================
# 4. COST CALCULATIONS — deterministic Python, no AI involved in any number
# ====================================================================================================
DEFAULT_INPUTS = dict(city="Lahore", marla_size="225 sq ft", plot="10 Marla", custom_marlas=12.0,
                      covered_per_floor=1575.0, floors=2, construction="RCC frame with brick infill",
                      quality="Standard", budget=0.0)


def sample_rates() -> dict:
    """A fresh, editable copy of the sample rate card."""
    return copy.deepcopy(SAMPLE_RATES)


def estimate(inp: dict, rates: dict) -> dict:
    """Calculate a full estimate from user inputs and a rate card.

    Plot area and covered area are kept separate. Covered area is entered per
    floor and multiplied by the number of floors exactly once.
    """
    marla_sqft = rates["marla_sizes"][inp["marla_size"]]
    marlas = inp["custom_marlas"] if inp["plot"] == "Custom" else rates["plot_sizes"][inp["plot"]]
    plot_area = marlas * marla_sqft
    area = max(inp["covered_per_floor"], 0) * inp["floors"]           # total covered area, counted once
    factor = rates["construction_factor"][inp["construction"]]

    # Grey structure = material quantities x unit rates + labour, adjusted for construction type.
    materials = []
    for key, m in rates["materials"].items():
        qty = m["per_sqft"] * area
        materials.append(dict(key=key, label=m["label"], unit=m["unit"], qty=qty, rate=m["rate"],
                              cost=qty * m["rate"] * factor))
    labour = rates["grey_labour_per_sqft"] * area * factor
    grey = sum(m["cost"] for m in materials) + labour

    # Finishing = rate for the chosen quality, split into trade categories.
    finishing = rates["finishing_per_sqft"][inp["quality"]] * area
    finish_items = [dict(label=k, cost=finishing * s) for k, s in rates["finishing_split"].items()]

    contingency = (grey + finishing) * rates["contingency_pct"] / 100
    total = grey + finishing + contingency

    warnings = []
    if inp["covered_per_floor"] > plot_area:
        warnings.append("Covered area per floor is larger than the plot area — check your inputs.")
    if area == 0:
        warnings.append("Enter a covered area to calculate an estimate.")

    return dict(inputs=dict(inp), plot_area=plot_area, marlas=marlas, area=area,
                coverage=(inp["covered_per_floor"] / plot_area * 100) if plot_area else 0,
                materials=materials, labour=labour, grey=grey, finishing=finishing,
                finish_items=finish_items, contingency=contingency, total=total,
                per_sqft=total / area if area else 0, warnings=warnings,
                rate_date=rates["rate_date"], rate_status=rates["status"],
                contingency_pct=rates["contingency_pct"])


def categories(est: dict) -> list[tuple[str, float]]:
    """Flat list of (category, cost) for charts and reports."""
    rows = [(m["label"], m["cost"]) for m in est["materials"]] + [("Grey labour", est["labour"])]
    rows += [(f["label"], f["cost"]) for f in est["finish_items"]] + [("Contingency", est["contingency"])]
    return rows


# ====================================================================================================
# 5. AI — key validation, chat, document RAG (LangChain + FAISS) and image editing
# ====================================================================================================
SYSTEM = """You are BuildWise AI, a construction and home-planning consultant for Pakistan's residential sector.
You help with planning, materials, construction methods, cost drivers, interior design and building-approval procedures.
Rules:
- Be practical and clear. Use short paragraphs or bullet points.
- State uncertainty openly. Never present engineering, structural or regulatory advice as approved professional guidance;
  recommend a licensed engineer/architect or the relevant authority (LDA, CDA, DHA, society) for final decisions.
- Never invent prices, bylaw values, fees or timelines. For costs, explain the drivers and point to the Cost Estimator
  and local quotes. Only quote regulations that appear in the provided document context.
- When document context is provided, answer from it and cite it as [filename p.X]. If it is not in the context, say so.
"""


# ---------- key & client ----------
def client() -> OpenAI:
    return OpenAI(api_key=st.session_state["api_key"], timeout=90, max_retries=2)


def validate_key(key: str) -> tuple[bool, str, dict]:
    """Check the key by listing models and making a tiny test call. Returns (ok, message, models)."""
    try:
        c = OpenAI(api_key=key, timeout=25, max_retries=0)
        ids = {m.id for m in c.models.list()}
        chat = next((m for m in CHAT_MODELS if m in ids), None)
        if not chat:
            return False, "This key has no access to a supported chat model.", {}
        c.chat.completions.create(model=chat, messages=[{"role": "user", "content": "ping"}],
                                  max_completion_tokens=16)
        image = next((m for m in IMAGE_MODELS if m in ids), None)
        return True, "Key verified", dict(chat=chat, image=image, all=sorted(ids))
    except AuthenticationError:
        return False, "We couldn't verify that key. Check it was copied in full from platform.openai.com.", {}
    except RateLimitError:
        return False, "The key is valid but has no available credit or is rate-limited. Check billing on OpenAI.", {}
    except PermissionDeniedError:
        return False, "This key doesn't have permission for the required models.", {}
    except APIConnectionError:
        return False, "Couldn't reach OpenAI. Check your internet connection and try again.", {}
    except APIStatusError as e:
        return False, f"OpenAI returned an error ({e.status_code}). Please try again.", {}


def friendly(e: Exception) -> str:
    if isinstance(e, AuthenticationError):
        return "Your API key is no longer valid. Disconnect and enter a new key."
    if isinstance(e, RateLimitError):
        return "OpenAI rate limit or credit limit reached. Wait a moment or check your billing."
    if isinstance(e, APIConnectionError):
        return "Couldn't reach OpenAI. Check your internet connection."
    return f"Something went wrong: {getattr(e, 'message', str(e))[:200]}"


# ---------- chat ----------
def stream_answer(history: list[dict], question: str, context: str = "", roman_urdu: bool = False):
    """Yield the assistant's answer token by token."""
    system = SYSTEM + ("\nReply in Roman Urdu (Urdu written in English letters)." if roman_urdu
                       else "\nReply in English unless the user writes in Roman Urdu.")
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    user = f"Document context:\n{context}\n\nQuestion: {question}" if context else question
    msgs.append({"role": "user", "content": user})
    stream = client().chat.completions.create(model=st.session_state["models"]["chat"], messages=msgs, stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def ask(prompt: str) -> str:
    """Single, non-streamed answer."""
    r = client().chat.completions.create(model=st.session_state["models"]["chat"],
                                         messages=[{"role": "system", "content": SYSTEM},
                                                   {"role": "user", "content": prompt}])
    return r.choices[0].message.content


def explain_estimate(est: dict) -> str:
    """Ask the model to explain numbers we calculated. It must not change or invent prices."""
    facts = dict(city=est["inputs"]["city"], covered_area_sqft=round(est["area"]), floors=est["inputs"]["floors"],
                 quality=est["inputs"]["quality"], construction=est["inputs"]["construction"],
                 grey_structure_pkr=round(est["grey"]), finishing_pkr=round(est["finishing"]),
                 contingency_pkr=round(est["contingency"]), total_pkr=round(est["total"]),
                 per_sqft_pkr=round(est["per_sqft"]), rates=f"{est['rate_status']} rates dated {est['rate_date']}",
                 breakdown={m["label"]: round(m["cost"]) for m in est["materials"]})
    return ask("Explain this construction estimate in plain language for a homeowner (about 120 words), "
               "then give 4 practical cost-saving suggestions as bullets. Use ONLY these numbers; do not add "
               "new prices or market comparisons. Remind them the rates are sample/editable.\n"
               + json.dumps(facts))


# ---------- document RAG (LangChain + FAISS) ----------
def build_index(files) -> dict:
    """Read PDFs page by page, split into chunks and embed them into a FAISS index."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    docs, info = [], []
    for f in files:
        pages = PdfReader(io.BytesIO(f.getvalue())).pages
        texts = [(i + 1, p.extract_text() or "") for i, p in enumerate(pages)]
        docs += [Document(page_content=t, metadata={"source": f.name, "page": n}) for n, t in texts if t.strip()]
        info.append(dict(name=f.name, pages=len(pages), text_pages=sum(1 for _, t in texts if t.strip())))
    if not docs:
        raise ValueError("No readable text found. Scanned PDFs need OCR before they can be searched.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=st.session_state["api_key"],
                           check_embedding_ctx_length=False)  # chunks are small; avoids a tiktoken download
    return dict(index=FAISS.from_documents(chunks, emb), files=info, chunks=len(chunks))


def retrieve(question: str, k: int = 4) -> tuple[str, list[str]]:
    """Return (context text, list of source labels) for a question, or ('', []) without documents."""
    kb = st.session_state.get("kb")
    if not kb:
        return "", []
    hits = kb["index"].similarity_search(question, k=k)
    srcs = [f"{d.metadata['source']} p.{d.metadata['page']}" for d in hits]
    ctx = "\n\n".join(f"[{s}]\n{d.page_content}" for s, d in zip(srcs, hits))
    return ctx, list(dict.fromkeys(srcs))


# ---------- interior redesign (image editing) ----------
def _png(image_bytes: bytes, max_side: int = 1536) -> tuple[bytes, tuple[int, int]]:
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue(), im.size


def ai_redesign(image_bytes: bytes, style: str, keep: list[str], notes: str = "") -> tuple[bytes, str]:
    """Edit a room photo into the chosen style. Returns (png bytes, model used)."""
    png, (w, h) = _png(image_bytes)
    size = "1536x1024" if w > h * 1.15 else "1024x1536" if h > w * 1.15 else "1024x1024"
    keep_txt = ", ".join(keep) if keep else "the overall room layout"
    prompt = (f"Redesign the interior of this room photo in a {style} style for a home in Pakistan. "
              f"Keep exactly unchanged: {keep_txt}. Keep the same camera position, perspective and lighting "
              f"direction. Change only furniture, finishes, colours, decor, lighting fixtures and soft furnishings. "
              f"Photorealistic interior photograph. {notes}").strip()
    c, errors = client(), []
    preferred = st.session_state.get("models", {}).get("image")
    for model in dict.fromkeys([m for m in [preferred, *IMAGE_MODELS] if m]):
        kwargs = dict(model=model, image=("room.png", png, "image/png"), prompt=prompt, size=size, quality="medium")
        for extra in ({"input_fidelity": "high"}, {}):          # retry without input_fidelity if unsupported
            try:
                r = c.images.edit(**kwargs, **extra)
                return base64.b64decode(r.data[0].b64_json), model
            except BadRequestError as e:
                errors.append(str(e))
                if "input_fidelity" not in str(e):
                    break
            except (NotFoundError, PermissionDeniedError) as e:
                errors.append(str(e))
                break
    raise RuntimeError("Image editing isn't available for this API key. gpt-image-1 may require "
                       "organization verification on platform.openai.com. Details: " + (errors[-1][:180] if errors else ""))


# ====================================================================================================
# 6. PDF REPORTS (fpdf2)
# ====================================================================================================
GREEN, TERRA, INK, MUT, LINE, CREAM = (47, 74, 58), (201, 132, 102), (35, 36, 31), (111, 107, 98), (229, 220, 203), (248, 244, 236)


def _t(s) -> str:
    """Core PDF fonts are Latin-1 only, so swap the few symbols we use."""
    s = str(s).replace("²", "2").replace("—", "-").replace("–", "-").replace("·", "-").replace("’", "'")
    return s.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.title_text = title
        self.set_auto_page_break(True, 18)
        self.add_page()

    def header(self):
        self.set_fill_color(*GREEN)
        self.rect(0, 0, 210, 28, "F")
        self.set_xy(14, 8)
        self.set_text_color(*CREAM)
        self.set_font("Times", "B", 20)
        self.cell(0, 8, "BuildWise AI")
        self.set_xy(14, 17)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, _t(f"{self.title_text}  -  generated {date.today():%d %b %Y}"))
        self.set_y(36)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUT)
        self.cell(0, 5, _t("Guidance only - not an approved engineering estimate or regulatory advice. Page ") + str(self.page_no()))

    def h2(self, text):
        self.ln(3)
        self.set_font("Times", "B", 14)
        self.set_text_color(*GREEN)
        self.cell(0, 8, _t(text), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*TERRA)
        self.line(self.l_margin, self.get_y(), self.l_margin + 30, self.get_y())
        self.ln(2)

    def para(self, text, size=9.5, color=INK):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*color)
        self.multi_cell(0, 5, _t(text), new_x="LMARGIN", new_y="NEXT")

    def table(self, head, rows, widths, align=None):
        align = align or ["L"] + ["R"] * (len(head) - 1)
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(*GREEN)
        self.set_text_color(*CREAM)
        for h, w, a in zip(head, widths, align):
            self.cell(w, 7, _t(h), fill=True, align=a)
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*INK)
        for i, r in enumerate(rows):
            self.set_fill_color(*((246, 241, 233) if i % 2 else (255, 255, 255)))
            for v, w, a in zip(r, widths, align):
                self.cell(w, 6.5, _t(v), fill=True, align=a)
            self.ln()

    def kpis(self, items):
        w = (210 - 28) / len(items)
        y = self.get_y()
        for i, (label, value) in enumerate(items):
            x = 14 + i * w
            self.set_fill_color(*CREAM)
            self.set_draw_color(*LINE)
            self.rect(x + 1, y, w - 2, 18, "DF")
            self.set_xy(x + 4, y + 2.5)
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*MUT)
            self.cell(w - 8, 4, _t(label))
            self.set_xy(x + 4, y + 8)
            self.set_font("Times", "B", 13)
            self.set_text_color(*GREEN)
            self.cell(w - 8, 7, _t(value))
        self.set_y(y + 22)


def estimate_pdf(est: dict, explanation: str = "") -> bytes:
    i = est["inputs"]
    pdf = Report("Construction cost estimate")
    pdf.h2("Project")
    pdf.table(["Item", "Value"], [
        ["City", i["city"]], ["Plot", f"{i['plot']} ({est['marlas']:g} marla x {i['marla_size']})"],
        ["Plot area", f"{est['plot_area']:,.0f} sq ft"], ["Covered area per floor", f"{i['covered_per_floor']:,.0f} sq ft"],
        ["Floors", i["floors"]], ["Total covered area", f"{est['area']:,.0f} sq ft"],
        ["Construction type", i["construction"]], ["Finishing quality", i["quality"]]], [90, 92], ["L", "L"])
    pdf.h2("Summary (PKR)")
    pdf.kpis([("Total estimate", f"{est['total']:,.0f}"), ("Grey structure", f"{est['grey']:,.0f}"),
              ("Finishing", f"{est['finishing']:,.0f}"), ("Per sq ft", f"{est['per_sqft']:,.0f}")])
    pdf.h2("Grey structure - materials")
    rows = [[m["label"], f"{m['qty']:,.1f} {m['unit']}", f"{m['rate']:,.0f}", f"{m['cost']:,.0f}"] for m in est["materials"]]
    rows.append(["Grey labour", "", "", f"{est['labour']:,.0f}"])
    pdf.table(["Material", "Quantity", "Rate (PKR)", "Cost (PKR)"], rows, [58, 46, 36, 42])
    pdf.h2("Finishing & contingency")
    rows = [[f["label"], f"{f['cost']:,.0f}"] for f in est["finish_items"]]
    rows.append([f"Contingency ({est['contingency_pct']}%)", f"{est['contingency']:,.0f}"])
    pdf.table(["Category", "Cost (PKR)"], rows, [120, 62])
    pdf.h2("Assumptions")
    pdf.para(f"Rates: {est['rate_status']} rate card dated {est['rate_date']}, editable in the app. These are not live "
             "market prices. Material quantities use rule-of-thumb coefficients per sq ft of covered area; final "
             "quantities must come from approved structural drawings and current local quotes.", 8.5, MUT)
    if explanation:
        pdf.h2("AI explanation")
        pdf.para(explanation.replace("**", "").replace("#", ""))
    return bytes(pdf.output())


def checklist_pdf(authority: str, project_type: str, items: list[tuple[str, str, bool]], steps: list[str]) -> bytes:
    pdf = Report(f"Building approval checklist - {authority}")
    pdf.h2(f"{authority} - {project_type}")
    pdf.para("General guidance only. Requirements differ by scheme, plot and project - confirm the final list "
             f"with {authority} before submitting.", 8.5, MUT)
    pdf.ln(2)
    pdf.table(["", "Document", "Notes"], [["[x]" if d else "[ ]", t, s] for t, s, d in items], [12, 80, 90], ["C", "L", "L"])
    pdf.h2("Typical procedure")
    for n, s in enumerate(steps, 1):
        pdf.para(f"{n}. {s}")
    return bytes(pdf.output())


# ====================================================================================================
# 7. SAVED PROJECTS — project data only, never the API key
# ====================================================================================================
def load_projects() -> list[dict]:
    try:
        return json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_project(name: str, est: dict):
    i = est["inputs"]
    projects = [p for p in load_projects() if p["name"] != name]
    projects.insert(0, dict(name=name, saved=datetime.now().strftime("%d %b %Y %H:%M"), inputs=i,
                            city=i["city"], plot=i["plot"], floors=i["floors"], quality=i["quality"],
                            area=round(est["area"]), total=round(est["total"])))
    PROJECTS_FILE.write_text(json.dumps(projects[:20], indent=2), encoding="utf-8")


# ====================================================================================================
# 8a. PAGE — Welcome / API key
# ====================================================================================================
def _left_panel():
    html(f"""<div class="wl">
    <div style="display:flex;align-items:center;gap:13px">
      <svg width="46" height="46" viewBox="0 0 44 44" fill="none" aria-hidden="true"><path d="M6 20 22 6l16 14v18H6V20Z" stroke="#F4EFE6" stroke-width="2.2" stroke-linejoin="round"/><path d="M17 38V26h10v12" stroke="#D9A07F" stroke-width="2.2"/></svg>
      <div><div class="serif" style="font-size:31px;font-weight:700;line-height:1;color:#F4EFE6">BuildWise AI</div>
      <div style="font-size:9.5px;letter-spacing:3px;opacity:.7;margin-top:5px">PLAN · BUILD · LIVE SMARTER</div></div>
    </div>
    <div class="rise"><h2>Intelligent planning for the homes Pakistan builds.</h2>
      <p class="lead">Cost estimates, approval guidance, expert answers and interior concepts — in one refined workspace.</p></div>
    <div class="collage">
      <div class="a rise d1"><img src="{img('hero.jpg')}" alt="Modern concrete and glass house"></div>
      <div class="float" style="left:24px;top:205px;width:250px;animation-delay:.4s">
        <div style="display:flex;justify-content:space-between;align-items:center;font-size:11.5px"><span style="color:#6F6B62">Estimate preview</span><span class="pill" style="background:#F4E2D6;color:#93472A">SAMPLE</span></div>
        <div class="serif" style="font-size:27px;font-weight:700;margin-top:4px">PKR 18.9M</div>
        <div style="display:flex;gap:4px;align-items:flex-end;height:30px;margin-top:6px">
          <span style="flex:1;height:40%;background:#2F4A3A;border-radius:3px"></span><span style="flex:1;height:85%;background:#2F4A3A;border-radius:3px"></span>
          <span style="flex:1;height:100%;background:#C98466;border-radius:3px"></span><span style="flex:1;height:45%;background:#2F4A3A;border-radius:3px"></span>
          <span style="flex:1;height:28%;background:#DCCFBC;border-radius:3px"></span></div>
      </div>
      <div class="b rise d3"><img src="{img('interior.jpg')}" alt="Warm contemporary living room"></div>
      <div class="float" style="right:46%;top:16px;display:flex;align-items:center;gap:10px;animation-delay:1.2s">
        <span style="width:26px;height:26px;border-radius:50%;background:#2F4A3A;display:flex;align-items:center;justify-content:center"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3"><path d="m5 12 5 5 9-10"/></svg></span>
        <div><div style="font-size:12.5px;font-weight:600">Approval checklist</div><div style="font-size:11px;color:#6F6B62!important">Track every document</div></div>
      </div>
    </div>
    <div class="feats"><span>Cost estimator</span><span>Approval assistant</span><span>AI consultant</span><span>Interior redesign</span></div>
    </div>""")


def page_welcome(toggle_theme):
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        _left_panel()
    with right:
        _, t = st.columns([6, 1])
        t.button("", icon=":material/light_mode:" if dark() else ":material/dark_mode:", key="wl_theme",
                 on_click=toggle_theme, help="Switch light / dark theme")
        html("""<div class="rise d1" style="max-width:470px">
        <div class="stepper"><span class="on"><b>1</b>Connect</span><i></i><span><b>2</b>Validate</span><i></i><span><b>3</b>Dashboard</span></div>
        <h1 class="wtitle">Welcome to<br><em>BuildWise</em> AI</h1>
        <p style="color:var(--ink2);font-size:15.5px;line-height:1.6;margin:0 0 6px">Connect your OpenAI API key to unlock the consultant,
        document Q&amp;A and interior redesign tools.</p></div>""")

        with st.container(key="card_connect"):
            with st.form("connect", border=False):
                key = st.text_input("OpenAI API key", type="password", placeholder="sk-proj-••••••••••••••••••••",
                                    help="Used only in this browser session. Never saved to disk or logs.")
                submitted = st.form_submit_button("Connect securely", icon=":material/lock:", type="primary",
                                                  width="stretch")
            status = st.empty()
            if submitted:
                key = key.strip()
                if not key.startswith("sk-"):
                    status.html('<div class="msg err">That doesn\'t look like an OpenAI key — it should start with <b>sk-</b>.</div>')
                else:
                    status.html(loader("Validating with OpenAI — checking model access…"))
                    ok, msg, models = validate_key(key)
                    if ok:
                        st.session_state.api_key, st.session_state.models = key, models
                        status.html(f'<div class="msg ok">✓ Key verified — opening your workspace… (model: {models["chat"]})</div>')
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        status.html(f'<div class="msg err"><b>Couldn\'t connect.</b>&nbsp;{msg}</div>')

        html("""<div class="trio" style="max-width:470px;margin-top:18px">
        <div><b>Session only</b>Kept in memory, cleared on disconnect</div>
        <div><b>Never saved</b>Not written to files, logs or reports</div>
        <div><b>Replaceable</b>Disconnect any time from the sidebar</div></div>
        <p class="note" style="margin-top:16px">No key yet? <a href="https://platform.openai.com/api-keys" target="_blank">Create one on OpenAI</a></p>""")


# ====================================================================================================
# 8b. PAGE — Building Approvals (defined early: the dashboard reuses its checklist state)
# ====================================================================================================
def checklist_state() -> tuple[str, str, list[tuple[str, str, bool]]]:
    """Current authority, project type and checklist items with their ticked state (shared with the dashboard)."""
    auth = st.session_state.get("appr_auth", "LDA")
    ptype = st.session_state.get("appr_type") or "New house"
    ticks = st.session_state.setdefault("ticks", {})
    items = [(t, s, ticks.get(f"{auth}|{ptype}|{i}", False)) for i, (t, s) in enumerate(APPROVALS["checklists"][ptype])]
    return auth, ptype, items


def _tick(k: str):
    st.session_state.ticks[k] = st.session_state[f"cb_{k}"]


def _pick_authority(auth: str):
    st.session_state.appr_auth = auth


def kb_uploader(key: str):
    """Shared document knowledge base (also used by the AI Consultant)."""
    files = st.file_uploader("Upload bylaws or regulations (PDF)", type="pdf", accept_multiple_files=True, key=key)
    if files and st.button("Index documents", icon=":material/library_add:", type="primary", key=f"{key}_btn"):
        slot = st.empty()
        slot.html(loader(f"Reading and indexing {len(files)} PDF(s)…"))
        try:
            st.session_state.kb = build_index(files)
            slot.empty()
            st.toast(f"Indexed {st.session_state.kb['chunks']} passages", icon=":material/check_circle:")
        except Exception as e:
            slot.empty()
            st.error(str(e) if isinstance(e, ValueError) else friendly(e))
    kb = st.session_state.get("kb")
    if kb:
        html("".join(f"""<div class="mrow" style="border:1px solid var(--line);margin-top:6px">
        <span style="width:34px;height:40px;border-radius:6px;background:linear-gradient(160deg,var(--terra),#A9603F);color:#fff;font-size:9px;font-weight:700;display:flex;align-items:flex-end;justify-content:center;padding-bottom:5px">PDF</span>
        <div style="flex:1;min-width:0"><div style="font-size:13px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{f['name']}</div>
        <div class="note">{f['pages']} pages · {f['text_pages']} with text · indexed</div></div></div>""" for f in kb["files"]))
        if st.button("Remove documents", icon=":material/delete:", key=f"{key}_clear"):
            st.session_state.pop("kb")
            st.rerun()


def page_approvals(go):
    banner("BUILDING APPROVALS", "Building Approval Assistant",
                 "Document checklists, procedure guidance and answers from your own bylaw PDFs — with sources.", "f_approval.jpg")
    auths = APPROVALS["authorities"]
    current = st.session_state.get("appr_auth", "LDA")
    for col, (code, a) in zip(st.columns(len(auths)), auths.items()):
        on = code == current
        with col, st.container(key=f"auth_{code}"):
            html(f"""<div style="padding:16px 16px 14px;min-height:132px;{'background:var(--gsoft);box-shadow:inset 0 0 0 2px var(--green);border-radius:20px' if on else ''}">
            <div class="hd"><span style="width:42px;height:42px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:12.5px;
            background:{'var(--green)' if on else 'var(--card2)'};color:{'var(--on)' if on else 'var(--ink2)'}">{code[:3].upper()}</span>
            <span style="width:18px;height:18px;border-radius:50%;border:2px solid {'var(--green)' if on else 'var(--line2)'};background:{'var(--green)' if on else 'transparent'}"></span></div>
            <div style="font-size:14.5px;font-weight:600;margin-top:10px;line-height:1.3">{a['name']}</div><div class="note">{a['area']}</div></div>""")
            st.button(f"Select {code}", key=f"hit_auth_{code}", on_click=_pick_authority, args=(code,))

    left, right = st.columns([1.5, 1], gap="large")
    with left, st.container(key="card_check"):
        card_head(f"Document checklist · {current}")
        st.segmented_control("Project type", list(APPROVALS["checklists"]), default="New house", key="appr_type",
                             label_visibility="collapsed")
        auth, ptype, items = checklist_state()
        done = sum(d for *_, d in items)
        circ = 2 * 3.1416 * 32
        html(f"""<div style="display:flex;align-items:center;gap:20px;padding:14px 16px;border-radius:16px;background:var(--card2);margin:6px 0 8px">
        <div style="position:relative;width:76px;height:76px;flex:none"><svg viewBox="0 0 80 80" width="76" height="76" aria-label="Checklist progress">
        <circle cx="40" cy="40" r="32" fill="none" stroke="var(--line)" stroke-width="8"/>
        <circle cx="40" cy="40" r="32" fill="none" stroke="var(--green)" stroke-width="8" stroke-linecap="round" stroke-dasharray="{done / len(items) * circ} {circ}" transform="rotate(-90 40 40)" style="transition:stroke-dasharray .6s"/></svg>
        <div class="serif" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:21px;font-weight:700">{done}/{len(items)}</div></div>
        <div><div style="font-weight:600;font-size:15px">{'All documents collected' if done == len(items) else f'{len(items) - done} documents still needed'}</div>
        <div class="note">{APPROVALS['disclaimer']}</div></div></div>""")
        for i, (t, s, d) in enumerate(items):
            k = f"{auth}|{ptype}|{i}"
            st.checkbox(f"**{t}** — {s}", value=d, key=f"cb_{k}", on_change=_tick, args=(k,))
        steps = [f"{t}: {s}" for t, s in APPROVALS["procedure"]]
        x, y = st.columns(2)
        x.download_button("Export checklist (PDF)", checklist_pdf(auths[auth]["name"], ptype, items, steps),
                          file_name=f"{auth}_checklist.pdf", mime="application/pdf", icon=":material/download:", width="stretch")
        if auths[auth]["website"]:
            y.link_button(f"Official {auth} website", auths[auth]["website"], icon=":material/open_in_new:", width="stretch")

    with right:
        with st.container(key="card_proc"):
            card_head("Approval procedure")
            html('<div class="steps" style="margin-top:8px">' + "".join(
                f"""<div class="step"><span class="b" style="background:var(--green)">{n}</span><div><div class="t">{t}</div><div class="s">{s}</div></div></div>"""
                for n, (t, s) in enumerate(APPROVALS["procedure"], 1)) + "</div>"
                + '<div class="msg err" style="font-size:12px">Fees, timelines and setbacks change — BuildWise never predicts approval outcomes.</div>')
        with st.container(key="card_bylaws"):
            card_head("Ask your bylaws", "", '<span class="pill g">RAG · cited</span>')
            kb_uploader("appr_kb")
            if not st.session_state.get("kb"):
                st.caption("Upload the authority's regulations to ask about setbacks, coverage or heights. "
                           "Answers only quote what is in your documents.")
                return
            q = st.text_input("Question", placeholder="e.g. What are the setbacks for a 10 marla plot?", key="appr_q")
            if st.button("Ask", icon=":material/search:", type="primary", key="appr_ask") and q.strip():
                slot = st.empty()
                slot.html(loader("Searching your documents…"))
                try:
                    ctx, srcs = retrieve(q)
                    st.session_state.appr_answer = dict(q=q, a=ask(f"Document context:\n{ctx}\n\nQuestion: {q}\n"
                                                                    "Answer only from the context and cite sources."), srcs=srcs)
                except Exception as e:
                    st.error(friendly(e))
                slot.empty()
            ans = st.session_state.get("appr_answer")
            if ans:
                html(f'<div class="note" style="margin-top:6px">Q · {ans["q"]}</div>')
                st.markdown(ans["a"])
                html(" ".join(f'<span class="pill n">{s}</span>' for s in ans["srcs"]))


# ====================================================================================================
# 8c. PAGE — Dashboard
# ====================================================================================================
FEATURES = [  # key, page, title, text, image, icon, icon bg, icon colour, arrow bg, arrow colour, text colour
    ("cost", "Cost Estimator", "Construction Cost Estimator", "Grey structure and finishing estimates in PKR from editable local rates.",
     "f_cost.jpg", "calc", "rgba(255,255,255,.12)", "#F4EFE6", "#F4EFE6", "#2F4A3A", "#F4EFE6"),
    ("appr", "Building Approvals", "Building Approval Assistant", "Document checklists, procedures and answers from your bylaw PDFs.",
     "f_approval.jpg", "doc", "var(--card)", "var(--terra)", "var(--terra)", "#fff", "var(--ink)"),
    ("chat", "AI Consultant", "AI Construction Consultant", "Ask questions in English or Roman Urdu and get practical guidance.",
     "f_consult.jpg", "chat", "rgba(255,255,255,.3)", "var(--f3fg)", "#B8674A", "#fff", "var(--f3fg)"),
    ("room", "Interior Redesign", "Interior Redesign", "Upload a room photo and restyle it with AI.",
     "f_interior.jpg", "sofa", "var(--beige)", "var(--green)", "var(--green)", "var(--on)", "var(--ink)"),
]
DASH_SUGGESTIONS = ["5 marla ghar ka grey structure kitne ka banega?", "Which foundation suits clay soil?",
               "Explain the building approval process"]


def _ask(q: str, go):
    st.session_state.pending_q = q
    go("AI Consultant")


def _ask_from_box(go):
    q = st.session_state.get("dash_q", "").strip()
    st.session_state.dash_q = ""
    if q:
        _ask(q, go)


def _current_estimate():
    if "estimate" in st.session_state:
        return st.session_state.estimate, False
    return estimate(DEFAULT_INPUTS, sample_rates()), True


def _cost_card(est, sample):
    p = pal()
    i = est["inputs"]
    card_head("Project Cost Summary", "bars", f'<span class="pill n">{i["plot"]} · {i["city"]}</span>')
    budget = i.get("budget") or 0
    if budget:
        diff = (est["total"] - budget) / budget * 100
        badge = f'<span class="pill {"g" if diff <= 0 else "t"}">{"↓" if diff <= 0 else "↑"} {abs(diff):.1f}% {"under" if diff <= 0 else "over"} budget</span>'
    else:
        badge = '<span class="pill n">Add a budget in the estimator</span>'
    label = "SAMPLE PROJECT" if sample else f"{est['rate_status']} RATES"
    html(f"""<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-top:8px">
    <span class="serif" style="font-size:20px;color:var(--ink2)">PKR</span><span class="big">{pkr(est['total'])}</span></div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:8px">{badge}<span class="pill t">{label}</span>
    <span class="note">rates dated {est['rate_date']}</span></div>
    <div class="stats"><div><b>{est['area']:,.0f} ft²</b><span class="note">Covered area</span></div>
    <div><b>PKR {short(est['grey'])}</b><span class="note">Grey structure</span></div>
    <div><b>PKR {est['per_sqft']:,.0f}</b><span class="note">Per ft²</span></div></div>""")
    rows = categories(est)
    colors = [p["green"]] * 6 + [p["terra"]] * len(est["finish_items"]) + [p["sand"]]
    names = [r[0].replace(" (rebar)", "").replace(" (aggregate)", "").split(" &")[0] for r in rows]
    fig = pgo.Figure(pgo.Bar(x=names, y=[r[1] for r in rows], marker_color=colors, marker_cornerradius=6,
                             hovertemplate="%{x}<br>PKR %{y:,.0f}<extra></extra>"))
    fig.update_xaxes(tickangle=-35, tickfont_size=10.5)
    style_fig(fig, 270).update_layout(margin=dict(l=44, b=70))
    st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", theme=None, key="dash_cost")
    html('<div class="note"><span style="color:var(--green)">■</span> Grey structure &nbsp; '
               '<span style="color:var(--terra)">■</span> Finishing &nbsp; <span style="color:var(--sand)">■</span> Contingency</div>')


def page_dashboard(go):
    est, sample = _current_estimate()
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    _, _, items = checklist_state()
    done = sum(d for *_, d in items)

    q, _ = st.columns([2.2, 1])
    q.text_input("Ask BuildWise AI", key="dash_q", placeholder="Ask BuildWise AI anything — press Enter",
                 label_visibility="collapsed", on_change=_ask_from_box, args=(go,))
    html(f"""<div class="hero"><img src="{img('hero.jpg')}" alt="Modern concrete house">
    <h1>{greet}</h1><p>Your AI partner for smarter construction and better living.</p>
    <div class="hstats"><span>Estimate <b>PKR {short(est['total'])}</b></span><span>Checklist <b>{done}/{len(items)}</b></span>
    <span>Documents <b>{len(st.session_state.get('kb', {}).get('files', []))}</b></span><span>Saved projects <b>{len(load_projects())}</b></span></div>
    <div class="quote">“Better buildings, brighter futures.”</div></div>""")

    for col, (key, page, title, text, image, ic, ibg, ifg, gbg, gfg, fg) in zip(st.columns(4), FEATURES):
        with col, st.container(key=f"feat_{key}"):
            html(f"""<div class="feat" style="color:{fg}"><div class="row">
            <span class="fi" style="background:{ibg}">{icon(ic, 23, ifg)}</span>
            <span class="go" style="background:{gbg};color:{gfg}">→</span></div>
            <h4 style="color:{fg}">{title}</h4><p style="color:{fg}">{text}</p></div>
            <img class="fimg" src="{img(image)}" alt="">""")
            st.button(f"Open {page}", key=f"hit_{key}", on_click=go, args=(page,))

    c1, c2, c3 = st.columns([1.15, 1, 1])
    with c1, st.container(key="card_cost"):
        _cost_card(est, sample)
    with c2, st.container(key="card_mat"):
        card_head("Materials Breakdown", "bars", f'<span class="pill n">{est["inputs"]["quality"]}</span>')
        top = max(m["cost"] for m in est["materials"]) or 1
        rows = "".join(f"""<div class="mrow"><span class="swatch" style="background:{TEX[m['key']]}"></span>
            <div style="flex:1;min-width:0"><div style="display:flex;justify-content:space-between;font-size:13.5px">
            <b style="font-weight:500">{m['label']}</b><span class="mut">PKR {pkr(m['cost'])}</span></div>
            <div class="note">{m['qty']:,.0f} {m['unit']}</div><div class="track"><i style="width:{m['cost'] / top * 100:.0f}%"></i></div></div></div>"""
                       for m in est["materials"])
        html(rows + f'<div class="note" style="margin-top:6px">Grey labour: PKR {pkr(est["labour"])}</div>')
        st.button("Open estimator", icon=":material/arrow_forward:", key="mat_go", on_click=go, args=("Cost Estimator",))
    with c3, st.container(key="card_appr"):
        auth, ptype, items = checklist_state()
        done = sum(d for _, _, d in items)
        card_head("Approval Checklist", "doc", f'<span class="pill n">{auth} · {done}/{len(items)}</span>')
        steps = "".join(f"""<div class="step"><span class="b" style="background:{'var(--green)' if d else 'var(--card)'}">{'✓' if d else ''}</span>
            <div><div class="t">{t}</div><div class="s">{s}</div></div></div>""" for t, s, d in items[:6])
        html(f'<div class="steps" style="margin-top:8px">{steps}</div>'
                   '<div class="note">General guidance only — confirm with the authority.</div>')
        st.button("Open checklist", icon=":material/arrow_forward:", key="appr_go", on_click=go, args=("Building Approvals",))

    c1, c2, c3 = st.columns(3)
    with c1, st.container(key="card_ai"):
        card_head("AI Construction Consultant", "chat")
        chat = st.session_state.get("chat", [])
        last = next((m["content"] for m in reversed(chat) if m["role"] == "assistant"), None)
        last = last.replace("**", "").replace("#", "").replace("\n- ", " · ") if last else None
        body = (last[:220] + "…") if last and len(last) > 220 else last or \
            "<b>How can I help today?</b><br><span class='mut'>Costs, materials, bylaws or planning — English ya Roman Urdu.</span>"
        html(f'<div class="bubble" style="margin:8px 0 12px">{body}</div>')
        for i, s in enumerate(DASH_SUGGESTIONS):
            st.button(s, key=f"sugg_{i}", on_click=_ask, args=(s, go), width="stretch")
    with c2, st.container(key="card_proj"):
        card_head("Recent Projects", "folder")
        projects = load_projects()
        p = projects[0] if projects else None
        title = p["name"] if p else "10 Marla Modern House"
        meta = f"{p['city']} · saved {p['saved']}" if p else "Lahore · sample project"
        total, area, floors = (p["total"], p["area"], p["floors"]) if p else (est["total"], est["area"], 2)
        html(f"""<div class="photo" style="margin-top:8px"><img src="{img('project.jpg')}" alt="House"></div>
        <div class="hd" style="margin-top:12px"><div><div style="font-size:16px;font-weight:600">{title}</div>
        <div class="note">{icon('pin', 12, 'var(--terra)')} {meta}</div></div><span class="pill t">{'Saved' if p else 'Sample'}</span></div>
        <div class="stats" style="margin-bottom:0"><div><b>{short(total)}</b><span class="note">Estimate</span></div>
        <div><b>{floors} floor{'s' if floors > 1 else ''}</b><span class="note">Structure</span></div>
        <div><b>{area:,.0f} ft²</b><span class="note">Covered</span></div></div>""")
        if len(projects) > 1:
            st.caption("Also saved: " + ", ".join(x["name"] for x in projects[1:4]))
    with c3, st.container(key="card_room"):
        card_head("Interior Redesign", "sofa")
        res = st.session_state.get("redesign_result")
        src = "data:image/png;base64," + base64.b64encode(res["image"]).decode() if res else img("interior.jpg")
        tag = f"Your concept · {res['style']}" if res else "Sample concept · Contemporary"
        html(f"""<div class="photo" style="position:relative;margin-top:8px"><img src="{src}" alt="Interior concept">
        <span class="pill" style="position:absolute;left:10px;bottom:10px;background:rgba(20,20,18,.6);color:#fff">{tag}</span></div>""")
        st.button("Open design studio", icon=":material/arrow_forward:", key="room_go", on_click=go,
                  args=("Interior Redesign",), width="stretch")


# ====================================================================================================
# 8d. PAGE — Cost Estimator
# ====================================================================================================
def _rate_editor(rates: dict):
    with st.expander(f"Rate assumptions · {rates['status']} · {rates['rate_date']}", icon=":material/tune:"):
        st.caption(rates["note"])
        df = pd.DataFrame([dict(key=k, Material=m["label"], Unit=m["unit"], Rate=m["rate"], PerSqft=m["per_sqft"])
                           for k, m in rates["materials"].items()])
        edited = st.data_editor(df, key="rate_table", hide_index=True, width="stretch", disabled=["key", "Material", "Unit"],
                                column_config={"key": None, "Rate": st.column_config.NumberColumn("Rate (PKR)", min_value=0),
                                               "PerSqft": st.column_config.NumberColumn("Qty per ft²", min_value=0.0, format="%.2f")})
        for _, r in edited.iterrows():
            rates["materials"][r["key"]].update(rate=float(r["Rate"]), per_sqft=float(r["PerSqft"]))
        a, b = st.columns(2)
        rates["grey_labour_per_sqft"] = a.number_input("Grey labour / ft² (PKR)", 0.0, value=float(rates["grey_labour_per_sqft"]), step=25.0)
        rates["contingency_pct"] = b.number_input("Contingency %", 0.0, 30.0, float(rates["contingency_pct"]), 1.0)
        cols = st.columns(3)
        for col, q in zip(cols, rates["finishing_per_sqft"]):
            rates["finishing_per_sqft"][q] = col.number_input(f"{q} finish / ft²", 0.0, value=float(rates["finishing_per_sqft"][q]), step=100.0)
        if st.button("Reset to sample rates", icon=":material/restart_alt:"):
            st.session_state.rates = sample_rates()
            st.session_state.pop("rate_table", None)
            st.rerun()


def _inputs(rates: dict) -> dict:
    D = st.session_state.get("est_saved", DEFAULT_INPUTS)   # remembered between pages
    with st.container(key="card_inputs"):
        card_head("Project inputs", "", '<span class="pill g">Auto-calculates</span>')
        a, b = st.columns(2)
        city = a.selectbox("City", rates["cities"], index=rates["cities"].index(D["city"]), key="est_city")
        marla = b.selectbox("Marla size", list(rates["marla_sizes"]), index=list(rates["marla_sizes"]).index(D["marla_size"]), key="est_marla",
                            help="Marla size differs between authorities and societies — confirm locally.")
        plot = st.segmented_control("Plot size", [*rates["plot_sizes"], "Custom"], default=D["plot"], key="est_plot") or D["plot"]
        custom = st.number_input("Custom size (marla)", 1.0, 200.0, D["custom_marlas"], 0.5, key="est_custom") if plot == "Custom" else D["custom_marlas"]
        a, b = st.columns(2)
        covered = a.number_input("Covered area / floor (ft²)", 0.0, 50000.0, float(D["covered_per_floor"]), 25.0, key="est_cov", format="%.0f")
        floors = b.number_input("Floors", 1, 6, D["floors"], 1, key="est_floors")
        construction = st.selectbox("Construction type", list(rates["construction_factor"]), index=list(rates["construction_factor"]).index(D["construction"]), key="est_type")
        quality = st.segmented_control("Finishing quality", list(rates["finishing_per_sqft"]), default=D["quality"],
                                       key="est_quality") or D["quality"]
        budget = st.number_input("Your budget (PKR, optional)", 0.0, step=500000.0, value=float(D["budget"]), key="est_budget", format="%.0f")
        _rate_editor(rates)
    inp = dict(city=city, marla_size=marla, plot=plot, custom_marlas=custom, covered_per_floor=covered,
               floors=int(floors), construction=construction, quality=quality, budget=budget)
    st.session_state.est_saved = inp
    return inp


def _results(est: dict):
    for w in est["warnings"]:
        st.warning(w, icon=":material/warning:")
    gp = est["grey"] / est["total"] * 100 if est["total"] else 50
    html(f"""<div class="total"><div style="display:flex;flex-wrap:wrap;gap:22px;align-items:flex-end;justify-content:space-between">
    <div><div style="font-size:13px;opacity:.75">Overall construction estimate</div>
    <div style="display:flex;align-items:baseline;gap:10px;margin-top:6px"><span class="serif" style="font-size:22px">PKR</span><span class="num">{pkr(est['total'])}</span></div>
    <div style="font-size:12.5px;opacity:.78;margin-top:8px">{est['area']:,.0f} ft² covered · {est['inputs']['floors']} floor(s) · {est['inputs']['quality']} · {est['rate_status'].lower()} rates {est['rate_date']}</div></div>
    <div style="display:flex;gap:26px"><div><div style="font-size:12px;opacity:.7">Per ft²</div><div class="kp">{est['per_sqft']:,.0f}</div></div>
    <div><div style="font-size:12px;opacity:.7">Grey</div><div class="kp">{short(est['grey'])}</div></div>
    <div><div style="font-size:12px;opacity:.7">Finishing</div><div class="kp" style="color:#F1C2A3">{short(est['finishing'])}</div></div>
    <div><div style="font-size:12px;opacity:.7">Contingency</div><div class="kp">{short(est['contingency'])}</div></div></div></div>
    <div class="bar"><span style="width:{gp:.1f}%;background:#F4EFE6;transition:width .6s"></span><span style="flex:1;background:#D9A07F"></span></div></div>""")

    p = pal()
    a, b = st.columns(2)
    with a, st.container(key="card_split"):
        card_head("Cost split")
        fig = pgo.Figure(pgo.Pie(labels=["Grey materials", "Grey labour", "Finishing", "Contingency"],
                                 values=[est["grey"] - est["labour"], est["labour"], est["finishing"], est["contingency"]],
                                 hole=.62, sort=False, marker=dict(colors=[p["green"], "#6E8F78", p["terra"], p["sand"]]),
                                 textinfo="percent", textposition="inside", insidetextorientation="horizontal", hovertemplate="%{label}<br>PKR %{value:,.0f}<extra></extra>"))
        fig.add_annotation(text=f"<b>{short(est['total'])}</b><br><span style='font-size:11px'>PKR total</span>",
                           showarrow=False, font=dict(size=22, family="Cormorant Garamond", color=p["ink"]))
        st.plotly_chart(style_fig(fig, 280), config={"displayModeBar": False}, width="stretch", theme=None, key="est_pie")
    with b, st.container(key="card_qty"):
        card_head("Material quantities", "", '<span class="pill g">Grey structure</span>')
        html("".join(f"""<div class="qrow"><span class="swatch" style="width:40px;height:32px;background:{TEX[m['key']]}"></span>
            <span style="flex:1;font-size:13.5px">{m['label']}<br><span class="note">@ PKR {m['rate']:,.0f} / {m['unit']}</span></span>
            <span class="v">{m['qty']:,.0f}</span><span class="note" style="width:44px">{m['unit'].split(' ')[0]}</span></div>""" for m in est["materials"])
                   + '<div class="note" style="margin-top:6px">Rule-of-thumb coefficients — final quantities come from structural drawings.</div>')

    with st.container(key="card_cats"):
        card_head("Category-wise costs", "bars")
        rows = categories(est)[::-1]
        colors = ([p["sand"]] + [p["terra"]] * len(est["finish_items"]) + [p["green"]] * 6)
        fig = pgo.Figure(pgo.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h", marker_color=colors,
                                 marker_cornerradius=5, hovertemplate="%{y}<br>PKR %{x:,.0f}<extra></extra>"))
        style_fig(fig, 360).update_layout(margin=dict(l=150))
        st.plotly_chart(fig, config={"displayModeBar": False}, width="stretch", theme=None, key="est_bars")


def _ai_and_downloads(est: dict):
    with st.container(key="card_explain"):
        card_head("What this estimate means", "spark", '<span class="pill t">AI explains · never sets prices</span>')
        slot = st.empty()
        if st.button("Explain with AI & suggest savings", icon=":material/auto_awesome:", type="primary", key="explain"):
            slot.html(loader("Reading your numbers and preparing suggestions…"))
            try:
                st.session_state.explanation = dict(total=est["total"], text=explain_estimate(est))
            except Exception as e:  # show a friendly message, keep the app running
                st.error(friendly(e))
            slot.empty()
        exp = st.session_state.get("explanation")
        if exp:
            if round(exp["total"]) != round(est["total"]):
                st.info("Your inputs changed since this explanation — click the button to refresh it.", icon=":material/refresh:")
            st.markdown(exp["text"])
            st.caption("AI-generated explanation of the calculated figures. Verify with a qualified engineer/QS.")
        else:
            st.caption("Get a plain-language explanation of these numbers plus cost-saving ideas.")

    a, b, c = st.columns([1, 1, 1.3])
    a.download_button("PDF report", estimate_pdf(est, (st.session_state.get("explanation") or {}).get("text", "")),
                      file_name="buildwise_estimate.pdf", mime="application/pdf", icon=":material/download:",
                      type="primary", width="stretch")
    rows = [dict(Category=n, Cost_PKR=round(v)) for n, v in categories(est)] + [dict(Category="TOTAL", Cost_PKR=round(est["total"]))]
    b.download_button("CSV data", pd.DataFrame(rows).to_csv(index=False), file_name="buildwise_estimate.csv",
                      mime="text/csv", icon=":material/table:", width="stretch")
    with c.popover("Save project", icon=":material/bookmark_add:", width="stretch"):
        name = st.text_input("Project name", value=f"{est['inputs']['plot']} · {est['inputs']['city']}")
        if st.button("Save", type="primary", key="save_proj"):
            save_project(name.strip() or "Untitled", est)
            st.toast("Project saved locally", icon=":material/check_circle:")


def page_estimator(go):
    banner("COST ESTIMATOR", "Construction Cost Estimator",
                 "Transparent, formula-based estimates in PKR. Every rate is visible and editable — the AI explains, it never sets prices.",
                 "hero.jpg")
    rates = st.session_state.setdefault("rates", sample_rates())
    left, right = st.columns([1, 1.65], gap="large")
    with left:
        inp = _inputs(rates)
    est = estimate(inp, rates)
    st.session_state.estimate = est
    with right:
        _results(est)
        _ai_and_downloads(est)


# ====================================================================================================
# 8e. PAGE — AI Consultant
# ====================================================================================================
CHAT_SUGGESTIONS = ["5 marla grey structure ka kharcha kaise nikalein?", "Best damp-proofing for a basement",
               "AAC blocks vs clay bricks in hot weather", "Explain the building approval steps"]
TOPICS = {"Materials": "What should I check when buying cement and steel for my house?",
          "Planning": "How should I plan room layout for a 10 marla house?",
          "Costs": "What are the main cost drivers in residential construction?",
          "Approvals": "What documents are usually needed for a building plan approval?",
          "Interiors": "Suggest a warm contemporary interior palette for a lounge.",
          "Methods": "What are good curing practices for concrete slabs in summer?"}


def _queue(q: str):
    st.session_state.pending_q = q


def _clear():
    st.session_state.chat = []


def _answer(question: str, box):
    chat = st.session_state.setdefault("chat", [])
    with box:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=":material/apartment:"):
            try:
                ctx, srcs = retrieve(question) if st.session_state.get("kb") else ("", [])
                text = st.write_stream(stream_answer(chat, question, ctx, st.session_state.get("lang") == "Roman Urdu"))
                if srcs:
                    st.caption("Sources: " + " · ".join(srcs))
            except Exception as e:
                text, srcs = None, []
                st.error(friendly(e))
    if text:
        chat += [dict(role="user", content=question), dict(role="assistant", content=text, sources=srcs)]


def page_consultant(go):
    banner("AI CONSULTANT", "AI Construction Consultant",
                 "Planning, materials, methods, costs and approvals — ask in English or Roman Urdu.", "f_consult.jpg")
    chat = st.session_state.setdefault("chat", [])
    left, right = st.columns([2.1, 1], gap="large")

    with right:
        with st.container(key="card_kb"):
            card_head("Knowledge base", "", '<span class="pill g">RAG</span>')
            st.caption("Answers cite your PDFs by page.")
            kb_uploader("chat_kb")
        with st.container(key="card_topics"):
            card_head("Topics")
            cols = st.columns(2)
            for i, (t, q) in enumerate(TOPICS.items()):
                cols[i % 2].button(t, key=f"topic_{t}", on_click=_queue, args=(q,), width="stretch")
        html("""<div class="total" style="padding:20px 22px"><div class="serif" style="font-size:21px;font-weight:600">How answers stay honest</div>
        <ul style="margin:10px 0 0;padding-left:18px;font-size:13px;line-height:1.75;opacity:.85"><li>Document answers cite page sources</li>
        <li>General answers are labelled as general</li><li>Uncertainty is stated, never guessed</li></ul></div>""")

    with left, st.container(key="card_chat"):
        a, b, c = st.columns([2.4, 1.6, 0.9])
        a.html(f"""<div style="display:flex;align-items:center;gap:12px"><span style="width:40px;height:40px;border-radius:50%;background:var(--green);color:var(--on);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700">AI</span>
        <div><div style="font-weight:600">BuildWise Consultant</div><div class="note"><span class="dot" style="width:7px;height:7px;box-shadow:none"></span>
        Online · {len(st.session_state.get('kb', {}).get('files', []))} document(s) indexed</div></div></div>""")
        b.segmented_control("Language", ["English", "Roman Urdu"], default="English", key="lang", label_visibility="collapsed")
        c.button("Clear", icon=":material/delete_sweep:", key="clear_chat", on_click=_clear, width="stretch")

        box = st.container(height=470, border=False)
        with box:
            if not chat:
                html("""<div class="bubble" style="margin:8px 0"><b>Assalam-o-alaikum! How can I help?</b><br>
                <span class="mut">Ask about construction, costs, materials, approvals or interiors. Upload PDFs on the right to get cited answers.</span></div>""")
            for m in chat:
                with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/apartment:"):
                    st.markdown(m["content"])
                    if m.get("sources"):
                        st.caption("Sources: " + " · ".join(m["sources"]))

        cols = st.columns(2)
        for i, s in enumerate(CHAT_SUGGESTIONS):
            cols[i % 2].button(s, key=f"cs_{i}", on_click=_queue, args=(s,), width="stretch")
        prompt = st.chat_input("Apna sawal likhein…" if st.session_state.get("lang") == "Roman Urdu"
                               else "Ask about construction, costs or approvals…")
        question = prompt or st.session_state.pop("pending_q", None)
        if question:
            _answer(question, box)
        transcript = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in chat)
        if chat:
            st.download_button("Download conversation", transcript, file_name="buildwise_chat.txt",
                               icon=":material/download:", key="dl_chat")
        st.caption("AI guidance can be wrong. Structural and regulatory decisions require a licensed professional.")


# ====================================================================================================
# 8f. PAGE — Interior Redesign
# ====================================================================================================
STYLES = {  # name: (thumbnail, palette, suggested materials)
    "Modern": ("t_modern.jpg", ["#F2F0EC", "#C9C3B8", "#8F8A80", "#3F3E3A", "#2F4A3A"], ["Matte porcelain", "Black steel", "Oak veneer", "Linen"]),
    "Minimalist": ("t_japandi.jpg", ["#F5F3EE", "#E3DFD8", "#C9C3B8", "#A39D92", "#5B5A54"], ["Microcement", "White oak", "Bouclé", "Brushed nickel"]),
    "Contemporary": ("t_earthy.jpg", ["#EDE3D3", "#D6BFA0", "#A9825D", "#6B4F38", "#6E8A6F"], ["Travertine", "Walnut", "Linen", "Rattan"]),
    "Classic": ("t_classic.jpg", ["#F1E9DA", "#D8C39C", "#8E4F3A", "#3F5245", "#C9A86A"], ["Marble", "Mahogany", "Velvet", "Brass"]),
    "Luxury": ("interior.jpg", ["#2E2C29", "#5B4A3B", "#B8945A", "#D9CBB3", "#7E2F2A"], ["Onyx", "Smoked glass", "Velvet", "Antique brass"]),
    "Japandi": ("t_japandi.jpg", ["#EEE8DE", "#D4C4AC", "#A58D6E", "#55524A", "#7F8F76"], ["Ash wood", "Paper lantern", "Stone", "Wool"]),
}
KEEP = ["walls and room layout", "doors and windows", "camera angle and perspective"]


def _pick_style(name: str):
    st.session_state.style = name


def iframe_html(doc: str, height: int):
    """Interactive HTML (needs JavaScript): st.iframe on new Streamlit, components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(doc, height=height)
    else:
        components.html(doc, height=height)


def _compare(before: bytes, after: bytes):
    """Drag-to-compare slider (small HTML component, works offline)."""
    b64 = lambda x: base64.b64encode(x).decode()
    iframe_html(f"""
    <style>body{{margin:0}}.w{{position:relative;width:100%;height:440px;overflow:hidden;border-radius:16px;background:#ddd}}
    .w img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}}#a{{clip-path:inset(0 50% 0 0)}}
    #h{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#FCF9F3;box-shadow:0 0 18px rgba(0,0,0,.35)}}
    #h:after{{content:'⇆';position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:44px;height:44px;border-radius:50%;
    background:#FCF9F3;color:#2F4A3A;display:flex;align-items:center;justify-content:center;font:20px sans-serif;box-shadow:0 8px 22px rgba(0,0,0,.3)}}
    .t{{position:absolute;top:12px;padding:6px 12px;border-radius:999px;font:600 12px sans-serif}}
    input{{position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}}</style>
    <div class="w"><img src="data:image/png;base64,{b64(before)}"><img id="a" src="data:image/png;base64,{b64(after)}">
    <div id="h"></div><span class="t" style="left:12px;background:rgba(28,28,24,.65);color:#fff">AI concept</span>
    <span class="t" style="right:12px;background:rgba(252,249,243,.92);color:#23241F">Original</span>
    <input type="range" min="0" max="100" value="50" oninput="a.style.clipPath='inset(0 '+(100-this.value)+'% 0 0)';h.style.left=this.value+'%'"></div>
    """, height=446)


def page_redesign(go):
    banner("INTERIOR REDESIGN", "AI Interior Redesign",
                 "Restyle a room while keeping its walls, openings and camera angle.", "f_interior.jpg")
    style = st.session_state.setdefault("style", "Contemporary")
    left, right = st.columns([1.75, 1], gap="large")

    with right, st.container(key="card_studio"):
        card_head("1 · Room photo")
        up = st.file_uploader("Upload a room photo", type=["jpg", "jpeg", "png", "webp"], key="room_photo",
                              label_visibility="collapsed")
        card_head("2 · Choose a style")
        names = list(STYLES)
        for row in range(0, len(names), 2):
            for col, name in zip(st.columns(2), names[row:row + 2]):
                on = name == style
                with col, st.container(key=f"sty_{name}"):
                    html(f"""<div style="padding:5px;border-radius:18px;{'box-shadow:inset 0 0 0 2px var(--green);background:var(--gsoft)' if on else ''}">
                    <img src="{img(STYLES[name][0])}" style="display:block;width:100%;height:64px;object-fit:cover;border-radius:12px;
                    filter:{'brightness(.8) contrast(1.15)' if name == 'Luxury' else 'saturate(.6)' if name == 'Minimalist' else 'none'}" alt="">
                    <div style="display:flex;justify-content:space-between;align-items:center;padding:7px 4px 2px;font-size:13px;font-weight:600">{name}
                    <span style="width:14px;height:14px;border-radius:50%;border:2px solid {'var(--green)' if on else 'var(--line2)'};background:{'var(--green)' if on else 'transparent'}"></span></div></div>""")
                    st.button(f"Choose {name}", key=f"hit_sty_{name}", on_click=_pick_style, args=(name,))
        card_head("3 · Keep unchanged")
        keep = [k for k in KEEP if st.toggle(k.capitalize(), value=True, key=f"keep_{k}")]
        notes = st.text_area("Extra instructions (optional)", placeholder="e.g. add warm cove lighting, use olive green accents",
                             key="room_notes", height=80)
        go_btn = st.button("Generate redesign", icon=":material/auto_awesome:", type="primary", key="generate",
                           disabled=up is None, width="stretch")
        st.caption("Image editing aims to preserve the room's structure, but exact geometry can't be guaranteed. "
                   "Requires image-model access on your OpenAI account (each image is billed by OpenAI).")

    with left, st.container(key="card_canvas"):
        slot = st.empty()
        if go_btn and up:
            slot.html(loader(f"Generating your {style} concept — this usually takes 20–60 seconds…"))
            try:
                concept, model = ai_redesign(up.getvalue(), style, keep, notes)
                st.session_state.redesign_result = dict(image=concept, before=up.getvalue(), style=style, model=model)
                st.session_state.setdefault("gallery", []).insert(0, dict(image=concept, style=style))
                st.toast("Concept ready", icon=":material/check_circle:")
            except Exception as e:
                st.error(str(e) if isinstance(e, RuntimeError) else friendly(e))
            slot.empty()

        res = st.session_state.get("redesign_result")
        if res:
            _compare(res["before"], res["image"])
            a, b = st.columns([1.4, 1])
            a.caption(f"{res['style']} concept · generated with {res['model']} · drag the image to compare")
            b.download_button("Download concept (PNG)", res["image"], file_name=f"buildwise_{res['style'].lower()}.png",
                              mime="image/png", icon=":material/download:", type="primary", width="stretch")
        elif up:
            st.image(up, caption="Original photo — choose a style and generate", width="stretch")
        else:
            html(f"""<div class="photo" style="position:relative"><img src="{img('interior.jpg')}" style="height:420px" alt="Sample interior">
            <span class="pill" style="position:absolute;left:12px;top:12px;background:rgba(28,28,24,.62);color:#fff">Sample concept</span></div>
            <p class="note" style="margin-top:10px">Upload a well-lit photo of your room (taken straight on, showing walls and windows) to begin.</p>""")

        gallery = st.session_state.get("gallery", [])[:4]
        if len(gallery) > 1:
            st.caption("This session's concepts")
            for col, g in zip(st.columns(4), gallery):
                col.image(g["image"], caption=g["style"], width="stretch")

        pal, mats = STYLES[style][1], STYLES[style][2]
        html(f"""<div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:14px;padding:16px;border-radius:16px;background:var(--card2);border:1px solid var(--line)">
        <div style="flex:1 1 220px"><div class="note" style="letter-spacing:1.4px;font-weight:600">{style.upper()} PALETTE</div>
        <div style="display:flex;gap:8px;margin-top:10px">{''.join(f'<div style="flex:1"><div style="height:42px;border-radius:10px;background:{c};box-shadow:inset 0 0 0 1px rgba(0,0,0,.08)"></div><div class="note" style="font-size:10.5px;margin-top:4px">{c}</div></div>' for c in pal)}</div></div>
        <div style="flex:1 1 220px"><div class="note" style="letter-spacing:1.4px;font-weight:600">STYLE DIRECTION</div>
        <div style="display:flex;flex-wrap:wrap;gap:7px;margin-top:10px">{''.join(f'<span class="pill n" style="font-weight:500;padding:7px 11px">{m}</span>' for m in mats)}</div>
        <div class="note" style="margin-top:8px">Design guidance — verify finishes and prices with your supplier.</div></div></div>""")


# ====================================================================================================
# 9. APP SHELL — sidebar, navigation, theme toggle
# ====================================================================================================
# name -> (Material icon, render function)
PAGES = {
    "Dashboard": (":material/space_dashboard:", page_dashboard),
    "Cost Estimator": (":material/calculate:", page_estimator),
    "Building Approvals": (":material/assignment_turned_in:", page_approvals),
    "AI Consultant": (":material/forum:", page_consultant),
    "Interior Redesign": (":material/chair:", page_redesign),
}
def go(page: str):
    st.session_state.page = page


def toggle_theme():
    st.session_state.dark = not st.session_state.dark


def disconnect():
    """Forget the key and everything derived from it."""
    for k in ("api_key", "models", "kb", "chat", "explanation", "redesign_result"):
        st.session_state.pop(k, None)


def sidebar():
    with st.sidebar:
        html(f"""<div class="brand">
        <svg width="42" height="42" viewBox="0 0 44 44" fill="none" aria-hidden="true"><path d="M6 20 22 6l16 14v18H6V20Z" stroke="{pal()['green']}" stroke-width="2.4" stroke-linejoin="round"/><path d="M17 38V26h10v12" stroke="{pal()['terra']}" stroke-width="2.4"/></svg>
        <div><b>BuildWise AI</b><small>PLAN · BUILD · LIVE SMARTER</small></div></div>""")
        for name, (icon, _) in PAGES.items():
            st.button(name, icon=icon, key=f"nav_{name}", width="stretch", on_click=go, args=(name,),
                      type="primary" if st.session_state.page == name else "secondary")
        html(f"""<div class="promo"><img src="{img('side_house.jpg')}" alt="Contemporary house">
        <div><div class="serif" style="font-size:21px;font-weight:600;line-height:1.15">Turn your ideas into exceptional spaces.</div>
        <div class="note" style="margin-top:6px">Planning, approvals, estimating and design — all in one place.</div></div></div>""")
        is_dark = dark()
        st.button("Light mode" if is_dark else "Dark mode", icon=":material/light_mode:" if is_dark else ":material/dark_mode:",
                  key="theme_toggle", width="stretch", on_click=toggle_theme)
        models = st.session_state.get("models", {})
        html(f'<div class="status"><span class="dot"></span>OpenAI connected · {models.get("chat", "")}</div>')
        st.button("Disconnect key", icon=":material/logout:", key="disconnect", width="stretch", on_click=disconnect,
                  help="Removes the key from this session. You can enter a new one on the welcome screen.")


def main():
    st.session_state.setdefault("page", "Dashboard")
    st.session_state.setdefault("dark", False)
    if not st.session_state.get("api_key"):
        apply(sidebar=False)
        page_welcome(toggle_theme)
        return

    apply()
    sidebar()
    page = st.session_state.page
    with st.container(key="page_" + page.lower().replace(" ", "_")):   # new key per page = entrance animation
        PAGES[page][1](go)


# ====================================================================================================
# 10. EMBEDDED IMAGES (compressed WEBP, base64) — keeps the project to a single file
# ====================================================================================================
IMAGES = {
    "hero.jpg": "UklGRkSaAABXRUJQVlA4IDiaAABwrgKdASpMBEcBPpFCmkolo6irpvcMWXASCWds7gA1+738OM4fsedfIgIH5XsH8PKNu8S0qKxmXnzCj7LwiroJXfmR+znRC9ID6H2X4O+ndnfVat37Aezv//+n6/Wgh4BhiYLVg2zQ9SPDn8n+0/3n+I9HXO/3Caln0b9HemfTbwv+WuoR7c/4Hplfp93XuX/A9A73X/Cf9j/O/l16xWsZ8yagfDjUDvJn/3vMP+mf9b2Ev6Aa94RoWN//8RCQyshvLaKDzfBcPn4Kd6hDKf//uSQH/UpP+FUPXcKnHv+Ykoqi11Z6d52cXeqfwLj0qFryVjdvymgalnv62s0akHOZk5f0OUM8yp+0YINLQMVa1R7lxs+Y2vD+isb13sjdpC04ioLmTPtmwLQ2v+9klBNH2sYEex0fmqdY2FTjh/bTP1jT36KI0lJrNtA98iR8VpvzG1yxAW782Zd0NOxo37dKQRK1WXhVYi85C6Quq64JhgXZl6nydgK2Usjv1KSZPkTbeeb6aCYgnUrD9h0O4Dsfog+F286G9Pq0NjujJILBFszUo8oAQ5611c7n8CBCTea07f/ap//mFAuQUOys6IrwohDvGHv1kTT8qw/Yqj/+4AfAtn05EtyLLdehcXAEefwdAcmfwbpG8mO2N9uAzz+nY75d6CS9eOSz6UoRuBwIO7We2OQEi6+VOsOk5VOP55+dZUD0/RWFSC7o2p7GBSA3mOkr/a1L/oQXtywAdYvUwdYpOkEwdNp9Df74jFfgRxEE1Lehe+c3VKNvm3YJCCXT2UQpThSP/zKRIfP6NqxgfKavoQhOy/jyZ3uVUmcbxe5MphE/V6kfc6YZAkVNKKQX/7WV/79r/D7+ylIwWmDpWitSlvHPK/vWz/+6CL9wKuT3NFMO/4fvbsXL7S7fImRU/SeVDd+tRvP24WLCTuxw3Yp4w/akVc775Xb08FyJBM/ZG9IKicAan2jWeHAjz5vw7GJChZM7gFgMdb1D9i1nuanhQ4U7aKuTsACzaLse1G844Uyiwl2HZXs7pl1+ZgNCQ6vrZA6OUIEflvkKsdWbsoKRLhWK4DUEgoazyhTtpUDH9t9Zqba/69y/OxAF3G7hWX/qH3j//xtX7O4LOc4rhQu3i/pwf/+5W/v+a1P/RkUKuMOH+++tJW8P6tRfQEXTpuERFQ3WD8JLP/2z13or+aGYlfBSpAir7P33f/nGEJunwJMsn0taORGkid1zjXnC00GSQUEK568MeS62L5HMBcYdNTfz1SvycTGzdA2aui3sPBu5fztQNpp2iiCzrWi2iKhPm9m8o58UzS1pANk7BgRx7FPe333KFO+Dutvw1WUgLXWbVk7uIBFP7kR3o0nXP7vC//x0tx+c7Ca8FCdWePGVz+5R/1/23yIhPOb6/fQS2OeFz//SSeZroxL38pAhXOfTJwOVwdSh3Bwak3q9nj/a/KBK//9+dc4eUnf9nuNv+iVk37YGOKCB5Lv+gW0tWAIyF7FSfuHk38tySpk+LV78IaO/qcCyapW5D1Ubj8eZXY2nRQwm0SRQRVdSgGKIJgrmepQZGIYoKvyWCkaEXECsGx5gWagOVeOYfKnVGp62w13bAZ+5G3PSn1fKwg2SxxhY4gF741I/DArIMa80kuCL5Vi2NkukHC/qrgvNZf6cYz1FIGw8PbP/+jqWjOAnSqNoE6QP8BYZjuaL4XWrX/2lpcU2N/AxOWe7LUquIq3IHHyM/mquSlfHael909IQG9tPiDYQRxZmAtftHCJdY0g3xUQUBUgPZZRFojtuXrpyQl4ndre2eqDIL3jACbJWLksr5lwCaGO0uo9JmgjqO02RusLCoPZSlNZpIQGZGkSwA5zE23utZaMguVKwdKktzHdiOfobE0D6AdnmF+YNwCGW4KSKcoTVSANoOUK+J8sWpAp9O/N9u+gePZaWhFUftLGKSIZVmWpuOCtFvrU80pKUx0p2PVjfDRnTWoKDb77z8xisQfupQlmjh5+UjR9Dqbsz2pfNtC+xPjvH0KW3pTrwq6uU+yVgxbFk2NQe79TK5n+1tW0EA+ar6x+3zIHM1VeDaXWB5DuWevfmigBtiei2H0c3g7NzYvZBKvkqYkjDnk5Ec6aVEeFDa0/dPzcLQFFqKbcludvnDSDzzlLVKntelwNQe3swSoMO1EphKI+ogkvJqNOHqyqgf/jyYVvrepvZy2dl0D8YOJADousD4+5FBcc0X9UBc5HHqirYDwV6z+v57UH8HEXOyLCW5pWyENdWBTemrQ54MFhWuEe+dwqzVpQ7Na1GJuMtX2tbBK5x8vzM6qCWrglZNKXTIRnWzkl2xwWj3adeAibx0RtlGEAWqSnY6S+sP4i1WiKIsibFW198pp1PfxDza0vN3uDQPbsDIk03aX5lbXEdJ7inN6oWC9wXrtQwpdncaMaT3eeeoldUfDikx7KWqaYFSIXGDDWvX7Bkkxoj1HcC4rVLHQhl837S/Dp6bCqaeM6MDlGN+xSkSUAdnwPOA/E7KMKIfCMJOpvd4wtpW/Eg5c/FZWs6N3i+VYYIcEvG27w3jgv+fBNjT+AZqSMd6BUEjYbmfn2UPDp/OsmU2FigzZQrDnia9q5MY4V3EDgwExqk83LOwxmzFs+ojYY3orxkjdP3N96h0hr79JczwzDucsEH65mhWHgGHSmY/fLQ1XRk4j6CO8OSBBYo9/bppZeS5FouttzWnN/qgiB2kGHwes9uf+S60CEjG7zMQeJi45V9HIdTpr1OEvWmvBPn9aOeM8RDOb1FOf/cqEMk3dJfGjEsYJEsf1pGdcv1phEiMZJXsNc+cVSWdDqMA3uZvQRLLVjxAsoitE5pjkoETCYsrAgWRqMnqbX5R4l9i/BOw9V9p04G+kP01q0iF2HtmP+7f8LYJVKH8fjEcuQAVhTvWEMvcA8JQxliMZiXdDyTp1SW/+Ihps+WrYLKf+j2xoJhznDjEMa6Z1vJx8qL9YPqp1Q47EHir+PtMcsIzopOGE9gfSmHwqbEeUM2ywgjdkf0cx+BNe3k5BppBtgfgZBFwgmbEszcdJIGlz5/dleqIkQ3w7j1bnFVNNA4qbbp6k7tZtRbBPdT4RxJeT66rCPohtpnX1I8TqQAjKztnLCBYspzYTT/sLFawRCT59f1UixxrwxsSVC4+nMls//8jgDoC+CuBS+H0bjExhvsd6LAa7D/AhL3AWbrGQfik3QwIV8SEB3qgEKJ15emA30hRt8YMzL86MKJMN0k/aTFRf7ahp/WLgNwyFC7nN4Hx56d1++uRY1vuupQ08UWFS/cPjKrJb9cSkPXor26t4f5OdUWGxu+R/oElaWU7/Cj6xCUbVsBdGnAIKUFt9+GyJExMFtwVEIOxAEmy/27n4bUktEhf0TPDdFZAg+xTz7OIe7vfNBsMyKhPnsl8XCZv5fT2dN0Ij1ygDchOvXRy15Sent7h2T5zCJPjoofwPIeZmz58aj/9ioU98UwTBIqcSQbXrb1Q/nOURZJz/hz4jXzoCcqCZPOo2EPXcisK5mniDr7Jien+vVr5+0ywynwle4aSIwtEiLoejdKf07YFwIEMJPbcCBi8eAE5QzRu9ImWOiQVrgvOHcmJgynk3j5XbdzlB+4t1APaBZXVf1quS2Qg5ZelzJZuQbvwRAHkLCpx2RDRefjXoD/XBnEQguSIZ1rdC+Lz69W6hwOMSc94AQK8AWW7LYUEbF1t3XKV152FqbDsFi3ziHjGop9Lg7xSUjY55jIQ10xGUbxPSP5+oZYPvNc/KLis5szjFsqjXdT8CNwQ/wvW4P/FRhXhchRVT4LLK6Dn4hl9LCMIlY9RSp3N0u51nfdeK5JuAb1afD/NQeI3fIisWdtVDw5u4xj1Ve/e/TPBE98xDT89UuXaU9asojnn0btLjiQ1O67LqvuANfN5mdlbcpiIDJnt6GEh4L6xKgh0X6pOzB98Z0oywG4kINQWZaE/c4gq86hBc7JWZNKI5XFC9EOvC3diqk9wE5fhE9z49NOJAV5CKvLc1I491/ZNCSXmK/blb44Vpuz8jCkkK1LqXBgzzf88L71gnrX35zFVY0MGrLqWj0Mo2JzS+d/T4Ri1LKe1vCmk5mi6a5J53OUepy5+Q4tzgbYSDHljjRgZ38s8Tz+3LirrFtbkKg96UemLTDeU4dTc8uOHq1eElY4kJTd9+gpYXbI/0NkJk8hqylMgjAISsXgjeVkVnTehTZqUxkFW9tK+gNmndXFrhHVjBiPhG6FvDh8oQ/TOCFCH9JkyAxRY5UyLuEhzEpYHwEJJvMLOTkfbza7OMeW2aY0bx3nK77vJgPXfBZrWbqIlkiih6Z1UEpklSL9CLvtU8mmEwHBA3pegdeebwHy1urAqQ5er+vZdyKLbOg8+uy3+2TDOjzMNny9hZuCN/LnKx/TSwxvH1o/wb96rXvVyHLx0dRYR5KbwsTo14QYoot4jtd3PW2PpEuuzzwgPkoIoPzQrn7jQvg6ADF14oi+XY//9f1Esv7+gdrM6X6TP1YBKpFA9w3yFELJbpsVQ/z2QoZ+iEo7YU0syEAutDpofMNFpY6o3Y1/Nnmie+Z2GRh7qGkYY3ZNuEi0+qzA/NWLb01G3ys0iZUpUVMsv01dk5jT57wysBgB5BsNxvj1oXHoEV3EhEM5T8PcUqr9NDtNxL3g3+CrMenw0l63mTpu9VTUCu9ps8tu+4BKHGjtmyyix+5W4zeyHG6KWM+NTMptxbmhXoHYPLbXC8XXkB85nm9b+ammn/uvVQwIgno38uhDilRp1Udvl2W5JJh3q4ATDdpkK+oyRZ4QXZOnKaU2LGbQv4S1HJ8S9CVzO+FjMGa2Co+GgrShPuhjpovFajlRWioj1QGFhn+b387Y97el06gjxDxxrHGatw3e//R2W6T6ANMEmN2N7C4wBAD4SJW7E/9z0B15FoHzttA5NKrBExzyNtALg7JRmrRGLaOwjZRqRrRWEsiklRee4goGEjocK0VMCLjOeing8yPc8a00JLakJsdARf7QQSvWjwr9VIDbQ5dAXOVRIyhT9FTgRQb8dcI42dYw6U/0OSQqeFJ0mA2iePqI7BaZyguUJblQzpexSIaEwtdDYAafZuaCDRI28dFYkh2gN9VNealiwDSUK7sjjaMV1bG1ZXgWCpbusXRy15hioL/5LTaELlEUvSqgfT6fiEHaPguCs6wPHxn6JJYyBwIlKrVgXph7AeHhdvseWGqY4Ve3V4rMy3dVyr25i10bOym7NN1Wk3SU7aEz3f5X4AKoIItfX1sojfi8sRQYRAFhUSJ+QnlpLpuQ9c7rtd+1NHqxFQyAp43RwmNnkpUNI8etMvMrk+fflHJRNCr03fp0+84Eb/gRwg88LPPnQh5xelJYDrnvNrPHrmfW+mPCczLmcfwAyM1Vo1IvLO6Jdkf1YIhkeZoqHvXcMYMmpHD8ddXJkwu5dvg3o1lKO/TL0djEr+C4ym7aAksj/b9hfO52g+HIxgslG93XW2w/6hPSPWVqMVZJI6C3Ya3hHhFR0C8kQx1dHSR2AB9T8XiQEWqakJIlcdAk34BmTaR227OqafhvIJ8egiJIHrd2aZjQyskYfP9IQCEyz+36tyW9s9bdb61UzVKxGExZDVxUBk5RuNga0JPJhQoeDR5/gX7A+ZaOvydkOwKmMmDd65PFnhsqs5wiMYiYdjrJaR9Gs8oJrv9t8hfR8YEF6WqyoFy3kCa+mocnSiIALSWfF3FiiUGGkGWR/eyzjp9ZykiaHYS1FSU33MmQRTO0BXsWzfF7Vjz0UiODOHMu1+LNsqxyYc+ZyEQFSWd7Ms4fmR8P/ceur5faYqyL1m2R1ohCZJXNpHrl1PvGmLL4/esSzdlg3owDT3GPG01hYVRDyeHjatHFfcENOV7zWZhXsS8GR27i86uO9KG4teqkj3fXR5cJQ41iUr8Fbc3+jLfn6d8uq2MwOPcmX9rKdYOQ0FsMkBa6mXobMwFDjYtkuWxaJ/4UG8EVm8U5cYxZ05a2f+yrfks8tttzOEQoFaGF55ne/UmJF2aEWhgYRKPsXwru+1rpzf5JrRlw38N067gRBGqcm4X6iaE7uuSbmNsJ8jucD+aEw3/6K4Ku+pmHzngNga/dJ86eBsAG3Qe2TVSX5Vp7uT2mAfKIdXu7wH7VI65YIuc2TyKQ+Vj0sIVPuBRxmwf3viJDvn0M8/d8W64rFRiU/HZbf2iE1TKojL1vl2lFexCsiL8/t8SoX5tZAlrtjpJ9CTvkHmUcYWNah8xLpTcjhkTSE8abqwTIlGXDMPNC30vM1ivtNJgTWn0BRp2gj/IXVLVRnnLv2u4B4/uqLvPO0MOgCbqpwEcssps93FR+f6d52iNKv/EFF8qwBEUFfI5+pKYk4EhhneXkfGnpa3xwDua7iFFC2OK684WOjh3p45ikR5sQRu7Ibjz4oB9QBy2oUPLodfEQUYrnCzY7uBj5bcY2wE5eXrkfVLJR2upWZmxf5Cbgi4LedIKsz/uyeNiAHcR3cxDXac6ELOZ9JQeyY4FD23QEB+sII9pYZwwCNxqjBrhm2I1QQZytt1Utq0coTzyR3c4ZZVE1FkZ7pym3lv2fbtl5oEbNKK5Ai2dngOExvsTFqxesTd/mP+GKwjwTLdnc6FISgEVGLv3zbVFQSGtA+nawRjt3vsNVLxrizvglnBGkqKhrqho8rspSYsQForoRwLPimZqKBgVLafSYTNTOu3k2Mh3T16LGQq79mCHIfJMHeylTf5G1lKhI8z4MU0NMKw1juyCSdVD1Wq/zntBWIFmSAgyhBUjQ0F8qgBzSQ/sA80JS4l8ES6QfcX+PWGnnNpuxijmaHfbzvijWgtb7m3CIb4sZp1XLjLG237NT9qCJP3F3drVd5FLx6XljLlBRUUqqxOn41djPAfbwhPF1nMDXoeSDtXxRGfBX+4h17EDHO4DBKgHkeKenADlmZtGD7Q8CjACk85MAxaxYc2lJJAW49rVkcKnioIBBJZXzCsLKzy13fKdl9DUD4VP0bnl8bqVraoNkQh1d9M4fNO2+wo5U4CXMMCXIMw1FaXsXCepduaQQJhzNgSeoS3cFR3kktbRPSVsmFeapyIJSifHeiTSPIIIJuAsmySN6IsZcpdi/Nu0X9mMQ5sF1ebsLRGmkTep+Kiv+sRKVLzlOEUijoaIdomhmmPj4AzRdo24EmAcGf3b21uq3bXL6tYDxmunRUyn5Z8T/AMiwoKu+fLvd3BnknnrFpj6kV8mHZlU1u4kkOqR/llJ9kKAz5jA84ZRY/wHeR/PupcDUqYNnjU3jy2tibOEwR6F9+FNIlIvwmKeLUq97Pw7gIL1HRLgQAP7pNzd0dHBMFnOuzOBT+7BLMljq0xIgAmU7BP93r5Tb4F7UfhFegwE8JmABzRkL8I/8/jBpwwS54CHH2LGxoyI9ZbuWwx/HkebmVV88PfwmluEjpQWTzrwTs1QjPiAJfdX+U0SKin1CEBP8l/WmgpXN2EXqdho+O61WlrqCH/EWwYffC4J6lFwT9abV0BzDo+A3uyHVJa6bfJcMJch8bU8r5hkqI1itchSbrr6RbiXyJSRqJrAYiwGDbJLTUM3dFl2xSdelrmPZeg7sMIXMwl5n19PYJe1Hvw3nPPwlY77TUX53pW6RWG6V9uX3fqu8Skd1nL/vG3qtLjKpfhv+zsCEbVpvzGrf1K0GpW7s8u7Q2hOzMm8HhZNhmMPNhWbqc0TIjRhznFP1g6o4IxgQpucTf5EkBTqfkVaTRfd/PSsvP7VgOC0BNg4yFOKFR5WAxRbTpEjavuiCwJKrw27Ix8ET6R/rUjLrd0hSFC0QjYqgcVB7JoPOnXArrJlUSLg9F3ePr8rxrM/ZLRem+HgWtum2apL4aMq5iYVmi2M3xOB64Crm8rhmnqfP1tBox58Su+jkiyP8lyk3d7FLzz6eWr4lsOS8HLI9SKFzZGD8ngXaBguM7Pe7BZETf7A/HriV/QxikTOGXgoz4xghu4ff9A66+cL8SnzMudrRzXtkx/MHAvjZ9hraQ6hNkPREDxR4fLCwN7Ew7wKC8woM//IDzL9DO8tswYvfolS6hufdop49di+zfo8n85zOZRJdG2Byn1Ubvj8aDaR4rRoMZAj3OxwcSvCXAUxKTMqswMsJCxheWwuSxRrOeDv82A73LSvUjt4rerC5RdiC3+GQ3FVG3lAlzZ3ESNg+TQldEXH1Lnxql27kDYQt/VoT10P22Sh1WJO2TIZh7PoXPfJie02XvInJIlmNCcRkZOS0kw6Kcz57xyo+3X4pH4EnCKaZRn9X75rr1ogR5p9oimHZMa2TsqN4AEq8SAjk1fLz72AUgUQstRQRFknUuAqVrDhkQynVTWBwA7Aee3ARld4wNjO/gqjJ2HQa/BMUWw5tFbLnsI+1rKMBrWFC8Kkmnpat/VIDiMbd+RselFbEkh/NL7lOHXzVMufUF3v5i+mcnBAYb8xmytr+WOvLA2uv4fqTJMqdjMvq7n3dzFKe/CPs5nwjuUtyr3/94qd9Ph+mGcEhV02CrgCgnbEWIWM1CiLh0f7zngcC7WqIWsv6OfTOJwlk3dHFMa+xsuAVFKVANzZk79f9gf97BTurPcLxow6aiQI1ogcyqI5u8jnCy8yfAdB7/2gkS2kxDs+aH3tjPLeS6o+JqMqZQRfcPTN+PQbnBoMsqzKnuJ88BHrPAxhimdcfSjVxdaXEmSuW8phG1I3hlQMm+X8hv5qsfvZAVL54RwJDvdYhYkbdnOcQE0VyQ/NwOXe46CXUWR6yVOzhxfeIi2NrG0vh4kVY5/ChPd+n0zeA8e62OTr0iFj9s+NRznpb1qYqv164VNk2Qmni9ZHO2HMHi89sBmxQymwDb7MZl1WpfefKn9Gs4zdg+UrmwPdExpWJwM+e45Jh+VWt7+Pm9VrQzRprW0Oo03+fteU+zGMPSrgasqO/NIGfmIriPMm5mLY6fdEEKdsuEeynscHY2C2YXjRpv77BdF6NCqgdqOT0CZ9TrNIa2P3lwdr7jA3HuJGpEu1HqM6zQBBKsB+3IoGqx3ZVnmPSOkj4Od6i/GjfsHarPLnzIdHvS1Hi4xeqvMli+JyK8ci4akS5aJcBDzCvHUYSHsVV13IfM61nLEkVdJezFevm24goDaQaD+JQXvqZoteflah5qOGItXJafnTy++ylOVCckdR9krDaTgoK0bm/o/8VGE6TNTx1c5dPWiJuvmW1y7psL91Doj+sSbLhbRaq3GMUuJBXf3a+tGdtl/kpIkhgjcuc1PUwpPQWBXubLDR2mcLQBeLfEAbbuQrXGUOVwp3qHuOJW7gdR/IBAOse1fwzGkaJuOga40BM4eymWcXROy+0F/EItpMBOigjQer6QXVT6vFTSlr918DmQWgQQ4D2CO0xgADsYzA1tMPmQAADMALCBe0J9a6anC91xbedDm3NTRB03a9QpLN9Bx5uaZVPlTsricZt+XhzoxBNxaE1Uw2XQbf0Rn0L7ygAilxOwDf8ngRnAoEl7peIlhIqfj6SBkXw1v/dXGXTKPzEk44S7Xx1e5H8yqNYhwcAfBpng4ziJa6XZrku8Ya5dA+2hlOCGix1j4moAyIgVKo9rsjA5XA8c6qVbjkJwUiMXLp1DxQ0hMtZ93B0+YEdyDCBxVUC/Sbj1+D4XR+mu1dlhNIxlnyISM0mYlT+kK1lifLAcGt+rvz4YLSSutC7p7hkaR7CBfB0I2WqJ4tllTDUJApcaq0bFSWscCyKzPdwxe11MTf2rnLCs1ajp7hMBWvnRzxh9ieAhNCT9GxHEYWnYvXMheOzZPGE/xwHSbdREFOQhG35ToZ1nvCwaAVhMHoyuCI8hHkzqCdZQFyFaL5E2UupxyWUvG06QCzhXex4rBFZGwTGCBv8DcahTdQGV9K9cSKkmG1CRrs23NwlTjFIi9wy8JEnombcfwaEiLLdLHaA9XsZNUsaMTQf3nI9sDfOIDLK5hrzxwpVdWobnPmFZlW10hYVwCPmfImIPJfvdPM+VvY1bnAy2iKZYIk90Oiqookz8ivUxI8U7C/CXMH6Rs0E3pvH8BXEzSMuWUt1bAsRWqbmJfLAQlX2ROye7qPYSAVDO+HtiPCIx2+QVY9qBRrhNNBV85d+E5hISQVKnxZfHT8Afyl1hcH/Rxnas2nm2CfyIqn2SanGKijIHOAOMqhx4pBjCPZyiLaiH6/kyq5k9NpgyaWsezUBKPaaxVN9fOAjxEnoTxcBqLTKKek/HXaHt00ZMKt70+ROxRf9wkn/7Hz/dyXVwspf9VV4bYYi9n4jj9sfQPEDy7h6WYxZsHcTIbXR+3BlR2HJtgMPw6cdDPeM5ii+QxPlqdcmS170f9vdxdoThkdkQxu5bm2mT3+dX+1Oy/gdscXr0rlemlfePH1BqhrOBxbTvO3cjwCSRJhY5jDEYEn1Lj/LQxMQTCN+Z6S7PJ3RRSkQmxOPwV2V42OjmU3fA932I+XLFvzjQYcEUJbRyG6RlWrY/gVm+gIdf2L3JneM0aG+o6zy0ZStiS77T0b5m3tYWrRpIE+YYYBuJZAO+RwYQUPiUhP/1hEh2MZu3uMpkFb3XC655r4RwmIwJjQObsfjoOUGxykNNS7LraH/wzJWZKKgZvMB8X+kiAhcGOoP/MbB1jMe+JMyr43z826Oqvj9l1mPCWFRBcoXSTRDIgXVARgi6UDquvIFMtvjEX67SMW2a9CRahEoJA4yxIw3jou4n8M0yZcI5/td4e2TsL7v6IQGaGFZaQVo/zj0AC+eKZaFVPW5q8+eNZaCr/s5O/nsJC4Qd3fLrQJ8jlCuddGAfNbACXFumjBFJIUGQO4s67va9PZG1Rz3EvldqtOAOV66wvYiPUqrT1VCFeEbciI9DCpWFWAaN9+3jReLGGQewpiFs7bQik75IszwCqMXoyxpIPqY1TbzQfINB9rVPb3tWh/P21VYobndmg/sIIiVApBeOXpRZFApXVBEZqiGVDUWmE1bVyiW6XXhU/+K8RdUQfEFj+ZfWj1gLGYhqzlEcrdbsCblLMBYGL4y2sEXEBkOoAFJG1P+gZzRT+ZfiEAT4JI9Dn1Jr69zrXb9hW0+TEUWBMcvDPMdLjO8CcVvzgMqY1L0D0o7x8wrz/hY1yn4XDMybRI57q50BG5GnGmuWw7QrTl7HQVTLxAJg5rlrcMJcAG9xXcACDejxI4l7IK0jVDKFe8sK0bQRUaG54IMUtyqXiTRT+p/7AJ7/1YD0crfTUUisNAjyt2rXbkwVIbKxgYOejqBI7CxwYnUvbxAiMrGOLQ5cSbvnfpakvBkrOHziPBeX3CWcP4ta993E/nVWRGCpLClrMifXf60TwcESRI/QPJiBviL/w/uBQhaqkh8NqB8GAlZnuxiAu5DtDI71tBMIbhw2pIdnncZH3+BRh2bPdFkQCv/C/kzGRh60DT4DFqr/LJ3CqsUHBD68fOwvHOdwU04lzpc6+hCK53dQr5mCPI4omtN8SPVnN607v9t4FDV6BGAA35qgmc+F7f7ih3n3hpiWKE2aqNAdtzu/44/aj3n1ItQ3MvMPUpdY04Oqolq62l1lKrjSmH2Y3/yGuoh3cyIhllLMt/n9TmX3ail1AYDTNDSteHJFt80uTxvkQjK79YkP0q3B1wWXHSb7lEtUZEd9e7BBAXhK0cHMsqvf3kYJm+djFIeGbtIoFEkqJVZO9ZQ8p5VnwsISe9/rc9KWWVBBxcknUM7SJWNnnjiRBlx5T3XvV9iz/pVov+tA3h7VV0Aox2SeFkZf7+n70DBPXXtjo4yYmK950sa4kTpCQckbj1scTHFgupdNxyDd6I5F6VWsxtIoANSpMgFY4bxdvcvO/a/PaIC44CP1YmmGzE2ZUK8hHy/lAWrSHs/e4gGwZndlSFF23U8LvMCTfNh4b0UZZLXm5cwPJc9791cihjnDa7Sh6cLYfBr6CzOhoTlkPSob9RfyvRRhY+9O8H3iokHY//ANMndYcyf8dMuh3PTEIG5g5EcrE3WCBImQcbwspiQJQ0RPkWiYw+xbmAUplaR5SW9/HwHQPRwzklLSVTmZ891wZlrlh4Q2fWhMrJXYNZEr90zVs/JcfS+x/xVHhy4Hu4PcQf5CN1z3i2+k6n0s0WmJJw72HxKWruuLwRlgxh7ogDgwKo/rsBCXczuhYRU9SB+eokGpLE3oJzMrn8t69Gak+Mq1oi4XVRNiVxgoZip7wmErfdoAWc8nhSTw221s98pDD+aAxYaE/ln/t+WA85TMuOfQCPIUOPa9isLGuHUlR3ZCkI8DqdXAc7377Y/xKUmsAe+OF5gx0NC78BJsUQHySaSCefZvQrC8zyeez6KLF2BrZWpE6dKzKCnTHHSuw63IeU8aiDmsHrQeaGZpixinAoRVERMRuwoZT91UeXeltpPFFS2AeYy70Q6PmPUEwv94MW8RQsqhtCYe0zbR+Z+1TE79C4oTyB5OO4267p0fR1gTKZWdFfv0LXYgeHOc/voqxSkpKSHl4MLija3UvPz42Gv3KeeuXxCH5PvjvoOHp4kqxSxXtS60ceYhm0bMzJgZXTmECtjqxMDsYs2PsGKH+anOye6kKk2cRopj3riahYu9BJoVVkqGr8Gd+HCfgoDmrU22lwBT+jt9PQ5LzS6YyZL1pB5h1HwbOaDqAkIBspr4n4E/zlOB7HdrR+IAb6I5cFj570NK08EIrBvHP7FrBPuSFLIR0Cv3XKGcuXZWMwTszuOL9R8+SfqZN0arKooGGwCatp6Fm5rJ+pkPHkGTwUm7nwGxfFgngi+03KZtOH9EEayT0C8h9oeLcIhaDgKMfhaf2FjZFZgtyu36r8b6gq2aZtIcioT2bMfGHEr/ZxpnOm1Pn0tA4eNdOwsnY9vTtIvfBIO6/6u0Ksa2LPMRYeURm/fnkgDEsYZc0r88rAsJmurPgzM0Ikr7gGEizX+L49/mNIvzuinCABxxu62xkZ/hgopd6hiZOUjkJ4loMip2ju2X5/kl0bwD/Jw1m8Griw4CI8uQJzpbO4my+CkpYtEFLfcMQBZlhldYFkgHa8qNLHd11sxWmJTL5r5bb9L3gFxGhKcwhDUTShMbRp/bpY/ExJW3PNkgcl6xjG1FaiReR0TxxVsjhv0yXhcDAt8H4jN2A54P+ALlzdRVzrbpC35tszfNXiLQe2O885e6Ul8dPPorqahHmJ8jEmeAseDFFdEdWka3faOrUeieW2KHgT7GdDSNpcSS2WqDgwzJIBHlXbr7ZMA8IOE/I3Yqu48X1BHwgvxc+Vs0QLSn3hMTN+Dp8YiOaGdeeG5LcxE/a/VIZKxbVM+wqtrVDlf4YCjhjI7Kg4Tjj+cMQCky6VAAeAJSrOmwCr6MsN7hdVEntkk3Jx0iPo0Vnl2v7w1ldnVGPtwMuM74avn5UmPKDZMbSvJwQrCyynX3AtQ2xPLutC3qReARJbjvolrXeFMBAmbGTvLSkRjNrpQi+NX+O/jRCZ3QUUB6esBVdOnkss8UjWX5qJfN3uDHvWpVXLVS34hxY+8EJvM1TtUrgusWyv6NA1xoW8t/ghq6g0/QMC6yD7K0znCR+x6/E2YTf9fQwcyTcA/k3lhC5RgWWq7+kOFFYPER2Bk/8V6hxGbMZPfONw2xqUmVSUKrKei7YwBvFjpNoDHWFyZjlgSsir6Q6bydLNmJ44MSMDRI8dI5lejTSkQnafBNkBDrcrMP+IQk0vTmInIQ6mkX2aUTIbMaobCoANUAg+aAovH5emXf99lbDs6NXjKQO/2cKPZMPNa7woaCaWe0+JXL4hG+3I1fyftsawXvnzptYSfy5X3bbNOcwijf0283H5ZBr7SqIczZZv/q9sC80R3fWEh5NhYixQ+mkoWrJJHZxXfE2p8ueRjeee8fL9mMASov3oMNMSgDKB4+uowLWbcmk9lg42pxalfLCZS0RqTV20RgkgHyXOd8r9F4s/eDL2LQGvjHUKmyl5nM+SvPxL7b1e3k4HfhyY4Q4O9bcCl+bg+LohFLvBEuoeUg6nk86yAfXHwdmcsXdYblpUTuUUOsz1R3eyuWTyWnejfm5wtLNt21AyQYvn5JxeMXm14LDiMPwwJvZ4ayOcshSQyiE8weXCuascIS8KFL/Km1aXdd8SNEx9j37OxzQmB2/hWEoJwJnsQSM9O15RtcuyVXOJY1ol1C+FJ3pXU12Yg62dS5yGC/JJ45Z95GvNkDG7qUZfM+J1IxG+DriOiOiJl1QOHxp1TjqSIba5AfwCE9Sns9LTzsnH39i8hCCSc0NFifkVT4TzbJd4O7DHzwG28nQtrv3kNAKvjHSeSGKmAsQPBcXfSaucZL7uKk5wHtpScTnZs5u+DCEFlYT717cynUvGgHTcN6FmAmFYlqP1VHABz5gTtpESE8KZjfE49+atkuvt301ZhJTF2WCrSv10AfUbYldosT/Lq/wAuwb4sYsxnSf/SqU7G5y8vm+C0VAH7oaCGTY+VXzjtHI197umyk4a+aJUPXaBGYMi7QR42RqiyhGOIFw4WFIYVDd5hTzC6W40wGctqrSsEjciFvic8AAP0ZuhgNhbFd1hgo+S4ASbZFHmxzxXh2o4BriCmitpHQVwTqPPWMhZPsuA2GRSCX/aAwA65hoPuy79UunpbhOnR1S5nUSWRDyMd5b9Bcr4oG+Xgs5FlFRkplCgqmQjuWJOejgCYih+lf2255PzvdBR7IWaD1b1+l6dxWVXwpCKMgM1hZIgjUaHXIqIaND4N/CMnE6pyOu5S+SJn8BiW9VoHMDHk6G4wbHykk58icEXrqNiYypdokZ3tf/XTH3+lmf/ovvcWqMbIFjeX5RKYJfXZQG4UDhNkQkTewZI6qgHCHMtcswCcX/ng07LogDai/lApAQWN3PB91fT8yqLydadn+Af5tfIYJmE9nHwpQdpPTaBZbe/xlc0ca5zdbY4HgHIApGpjZGXigOutMqgCimkRtswf0VFCmB7FyBpQs022qEczNEK6HLulJ/74f0es+wtnpu03D6KFNeKdccQk5YBFoGLI8Nm4X5azQDmpzCuqXglnl+063BeP2njqAhebfgiqk1aOuCAIruH8f3NpIJDN/2cE5gWNxwV+7wHWIwp0DhsvbJmhQo8jKYPg4cpfE9LGyvEdd9p6IKpvU0za9HZ0KVNqlCxfiNIuypLZHC5CZ6S2IyGfsiF1Srmo5Io0wxHpTIytgfuQysRlE05ix5ILhlPy8ynJTufpcHz6BCg2cB/F/jyu2ByoRnm3n5aZBO8vOAMO3DU+/AXbF1o81jsBaPuVFF+GugojS1IiOYFccmItPYRbFIAKeqqB2ycAiagT13amZBTusE3IHS16DLEDsahxpMuEiv5wlorzpc2hUjYb0eIc+Mp4weqgCCGlhW6f4fU5Hh7haD8h5ze1IDeOk2vVdDav4uI4MHGRStvWLi1HqcFqPI7QQCOgyOIncAN3F7wQWsijTyn446JxHIY6v5fLOUg/NgqfLDV52CuuGYOV9bAk7zmWDCAyd1+hgY3QgxiYyjLgOuX1qcvNKyz+IQyFp+H5eGwMcWvludA2G+0K3MQ3FZKl03atwBCAIYQGErYXXb3IK6UkKjB+kj2gb1nOhcF9rngs7o+1y2u10UkDQOI4BqAYvbQlqGDPgeFwSm4tjCrCRsqO8LWfVt4KOA2mrkRvPe/B6ZcHsP/LGcmn0w7Fm4ZaVFFUwCZXxA4rRiJ8Xpp5xHGFfxIYM1G7qsELrEaHv1cShNsKyK4x1yGMkrA4OSJmAxxVOFw/p7whmYqluCq3kOOCJg8xq8zNltOY7HoxI87uD+jZ2VyB1KmukDAr5eRcqxiqL4VeBxm1+LMdhpGJKXBxv7UFZJtZmQTo8Z4jQAe0/38y2VFYV4vxWO/E+yfiaIpYdUiGNqkUAc6tTcUKL9vM+pCfbzQAX0Y1Upi9mLdq/UC7HUfi+7PobI/GN0OExBbe00eY74VCeBF2872CdpSuBmZwBDxe50dW0PPdZe2+DACoVlmcB1JPzEPWJFbL42fAekWFDj7scSBNWACO7pWg7FcTlXwH097sJQtuSSYeT5LMBRZQqmfhn22YbEepiYNX6juHS4xjkfvlV4dDXTBZ4GBQpZImXmeoV2EHcWmN7XkYP1KnCaZsLQM8/B71mCbhqtl52+lkaRN1B4i5jAGdsfxgLpjYcMDRh9x1POp2wI6AWhZHIOGNM9vq65Rit3oV9I2bYRnKC54cS+ApdYDL6RpB8VBUXVNagYHm5bP03Ck2i6aCcuU4j5CEuMkj+KZbNy7O21h/kTGD/KD7qyf5nw8v10NuGe+B/Khcl4kh1QpuvVdtoBv5U/YFEMRvauyG8hGSeksKF3gYxRszIBm/EcPQzg8vs9ook0DchdONsnnDdGJbBVAyOOFwVVjbYSkrsME5QzwWxSaGhJgyHSNHyNU6amynRoK1wapl9q9vEmDwzg+QO4zhOILWmQCr0zWz1KHam5Bm3uggzlE9nInsHiVWf0bm5ptinP8TvZQOZVbPNiCjqmDWBuBFKttfOn9rzxB0MXUCzHrf3EaHaRwMEFykodrWirzzBxqxRL1SchZeCwX2GCWbec9YO2nyUu+LEDQORByrDunDFhg7XO3TZywnXyaE5d3Qw8KGBi41ZMecmVD+uRHaQFRpr5fm0bixReFWKhLXHRs/YpwmcZrkoZEAW6xzogWquNkEdMyQvW0daOG/Gq6sDnO1GFw7V+yyJxmuD4MpYf7XYhFiE0IRhBBJRmes7CPc3T2fGmXOScbA7fAU4hS0UbVhW6oBIDLVvpHi2ahHtQnwEqr2/9RB56zRc8GvuDgCrgaHQ7nH1iYoI+dro/Acx/fv46/t9wBTQ2aOtpptNtDE4gifFiitkt03IsXjBIKGRMJkg98aDxeAfyxeN3sNQ6wGXvMxcIo0z1BwUlwqAdDzqA0ga7xDWeML/FZuXnqRN0wPiO/Hf4nwkO/QwBdhHCB6iymmN60UScFqMoknevaCoQAIQohGL78y4Nr49y1N3ut4HiBFYeBbp1NhadpJ506gZDVS2adpG6QHVfkW99/t3j2/m1aGkIt4QO3NIajN0p+t3OovtFByn3R6U2RecjIQpBWpQkcgVLSVsI/zrbFQxKO4+Ky4yP2i84aEc4zNE2sjtDxs/wyW96dB3mK3zQyWL/UiAoeDDbkdSCoLyhDAdNGmAL234tIcT374bIV0WzL0c5CJ9lb5KJUMRqb1F373q5o9lF+JCR9KR5vQ7+IBvtt5tJxdjezSjr7vSJ+gv8J5yUxo/Kj4lT7S8vsdA9zosW5lTKfq5u59gBmw2z1MlKaSpogh3IOTy1DzBRW3HD6rgHbL6MzVBZr0EoivW95uOa4CFfkJZVfX8WH+yQcFAN7Ued2fygbB04vjXCn6jC+R3hh0dv+PAhuPVrAHGuqDWyUX0ANyE0kadI1b0Df2bsAl19mvsgILrRNR1sZqj+CseepSCOiwHOqlkVXDxwfkob7s84wi/6Hx0WwUV5UYSWsQzL99B/XAzJNzIruIeKGrDEpjG/lsjLEq2jFx7JzvR9Y1l0xzAEVkRz8iMWFmkC4vDww1NkUPuGelPbv2g43+Ektq5lDh4HRT+01/2I9CQyLaP9AGqdr8BZypC6NbDZBl7qGLQfHC43cAlfEmQ/6F3iD4pqub0hz0QAiz6QiNxPHENWGHFkqStvp0jdz4fXcx/lRLFmoNqY9yAFuWm2mxWRS0KiqOC86ZyQRTr1VH3ix/TBocM+Kq5aLTleDtk5lAA/ytstBsEuXYnVGxgtRluJe2em3Xv9oTJdkCKvEeuML/zDCzas5p+h/9shF96rb8XmGpyhr5970c6S9rqXERaGvNRGQMh/i7hZFLvJ3x6oPgEoDPZ5NGmSq/xgetqgtwPCe0EI3O9/4WVcxgVz28QRdXwneNhC5ol2fn37G/HM4YsHhI+Y6aDbbUtMXEYkc5ranMOE2q9FZm/P8ZSWlh2t5JEGScnpzmjmyeqOVcJSks/8PQqyKHS4+QJBV4w+8WX9IHl22Rw/k+Pevu0Qo8PLQOEVp7+rxeSUo53zLoGJBwXeDzi4GZgSL6b0/NMeZb/hsHxQrk4/SYJGkNn7NIA+D3tqqu+FOHZ6VUPp/1NIts2zC/lkENKmjZk71pPj+uNNZFDy/FkbwwzMOWJZyTAl9bEgiZybvFzT/5WRBQnzdqGSGdtwAg24ibpJ+kx4cc7ch60hMW9nX1ZM9bhxomD/d/wvK3mzkLJDRJ4NglJa+LU8s7iW9Pv7Hd9ZHUgEaYfqjlPmKxzNkuG9JR7qucx5i1Ykl8ADFO0E9L/3PukC9nVWhNMf1FR1ELKw7miJ7iMesVlOqCe3YkTg+I1uzYtmP3kyWvWiS1CeJIRNGGH2jind/jfayZig5vQ8fjTwl3vWsM0+g6gwRlC4byIR2aWXSvP9PSdLWEc13DFKlmoIf0TtmFvmz7qps78VBZP2IBIhmMxQwWnmtV/LMIxCguLJOYeTss0GydoRqFU2TBKo/0+Z8e2qu7oYkauZ1dhiBWEnZ8vonkHkb09cotkJaBz+MzGFNexg+LURIz3D4gkuYiUAvy3V9L227nWNEXUAa/T7TkWYkUho0Q8aUNVBvnT2csU5hHdCUHnIBCCBfWLS2MCi9LeUgDrDT39SqeFFwvFp7KcqRDYSsLtbZwtYGdikzV2EjlFTAWp+0X3ZSFYaabfviXKfDOnfuhwPK2Up73+CRIj1YwlIlZ9TaTNW2q2AveVGP5FxQ9ltamHmVRyOKzIp8WFk2N5BL/SIXmHfg0mBmsus++tDZa7aKqoxBaELLrlVAJir65vuZlcsmJsfVKF/eDCSAzK384+/YCwtJCBcYRIKYsdNAaKQiCYqkFR+9Od38pu67pT2NCqZEP4y3UgTJDY5a5jUUHY1GsuWShZUncn48c0m2Nt4jDyShAPExC/9/Mlemk3kEDDqaZVzQjd7gIMFMhc6vsRdOH6J6UEBsHPAVHwUef3cKsCQbG5Z/wB1mK8BitzbK7ghs3J7yenEgl/7QxMBBUFbxDgBzBOwgVRRg21MoDMsA4qMC3BOdPhLzNzpUffDgTi7gKkKYYgaotUx6qJkwXWOCM3nmpgbCz4RgGiffi74lyXqQDEaCCsxpK6XF6hJTJdsjDOyXPSNlJPKuaf+vB1vrHZcEem7y7+k9dpDWdxYk/hKWFISvjyUCVezvZmCgLb0o/cvRBe88JudGH0GUd8fglk1G88oqd9JRQ/Vz/kS0R85hkxP7jowxuEiQ4PIt1VBqgmqmT7+/buR4t6wHnRHTpc/erlYdW4UoiPl7f8TxIYoUGeqrlUkZpInxJBvx9PSV+225eomWx2nT9RIOs8aQNG9g37k7vUCg5Q0JiRK33Oe/OfFdHa2Vv+ZifhtjxzR4AXVPQ0eEFVVumsf9y3O6kTWWERgV0T15V1PST/xfPR2JJphLiqwS9vKT+BGMc49G0yJYgeOkAAiTFIJReS7nALOxh2I1vGqkOc+GHY6hffA4G8oJds0U4dj0XmJb6W2ZsbgK0ZJKBROsg+93qyHdk1k+PWJeM2V40g9Tv04Aj0JDSJaDB+/5G5nHg5gzT2atrhw0IbwPW+/4bF1Pjot55IQkCHlzutc8TaevQ0s3j+Fk9uU3CtUi6gKPFHbeU89gfsgsCyBHIukadjMU9N7D3hH2/ugHLe5O+z9eXqykLXMt2ixLBwReyPLbmaAQQwIj6jaL37t0N5OjUeMBOFd5FNDIJ2R/iqnpvELD89kXMbrIvWQNL+AbXbuXrPmK6epxS6X3lDcCMftJr1HCBiwPayxjMwWc0BZSVQKaAUHcIdsNs07iilHdLp0Ze6A8SIBP7qMGLapzDHwHV1W+agxD7tSgHQ5J+ylbdWe4oXqP12PwxN21b2nqPNLI+C1dyFztRnDGClM4xIJeMDIWtqJvGsB/IenMsMHbTEFjX6izH9FuV//eJ7GhMoiaAzYdidMRmvKK3aIRLHbKLv8QJDoHzNGs3i65o5a39E1fhOP4M4wQFWJWDITAA6XPcZnrDymd9TCTJ0TpWZrW7rsKoROfzNBfwFmClpGuUhn/aaG+CKdkYO1FHks0KRgzNi5LGhR2s1iJIyqLEtiDVcOMgypDz6SN00Kw6Cf4EldwzpQLK+T0K7qlV+AA7WDeLt5YlI6WC5Q8LrmULOu1UaDZQjZeP13e8Bh7UmKUPGs2cOLcVKCAz8P+pyYwH6RU3ItR8qfbbfxDNfbAFetLaZY2bIudMk3dVm3au7S0BxdPrV/u765raKPnBXKKuo5e4LnMjSzWkYcgyBSoYQND2us/7pWZLkPjel2eoonzRf4//gV7J9GR+QwnrWwQXR5d+FVuAf3CnAYgyY7bZVz2WMMj+WpEIji8ei1IU1+o0vQQz7fVtqG5KzpWoEVhgFZ07RaMDk3fQ1hccPzGV7Lu5CXRqgEOW0lLnpaBmNnBvBQTV5RhBYwCrQ4oXGZELsNpK56VxZMnAMc+0busS3M8O2rqR6S8EvVooAHhvp2dbOrf9Tapktdeamb6EYvrgLDbetVHdJx8J4XRnDGoa4eNuQPtM9h89Ltc1yIfIDwzDbSoaR2Ji/66kZC7emvtRprEukEI9UPD9TglfDJHZIWYF73i+nLzoxBNpR+ZIbJGXeGM03HqlGpsK//bpyiuGDl7gCstWCahMXRlqabPMZUg8AMAJpuYsdhl6DLmW4tlQbGBpepjNzLKTWhQ7xKVDRBvc+5Uv7os1I4eJA+6S2ovTXf/ZY5PluUcKveFeoXI5AMwpuYlRVEqHKHXJbqnDySk7vOqF30loTV4eokLGmFx/nFpTc9ao0KPmBj8bPa3DkZvCCxxVkbMMLPDbnMv/+7jShKy3mCJscOilUdxCcM5+IuoyhubkdhgYC4wu03fMfT8X4Iq3ceFVDGREDjmtoNjJJoFDbJCW6tOX5OU33NrgKwOkEPW4EfFlL1i2Cq1vmMAk8qMYg2d4Jwo72chXJ3qw22XR+WN9w3vQzO8/SoXzW1baYJjkPIAB9T6vvjAn7CcBGflSqYjEWa/DeL6eYISM3GufTPhx4/dw+x7Eg56GXC80cFKfVIevE9ZF6CCCfOp3UgkeRIWB71b3egEpNJp1mZXu6PuMwDPIEemJamcbtdZ9p1DdJx2vXFecfuq3aGzypcvJ6lNg6JvsRV5FsOL++WAIPB5ShG3AJjXOVmRx5A8DTrd75jeRyb0odSck00/2cv3zWGJ5Psn9WueYvfDUkqoZtRms1kkuB/QYlOcoJmosS0LFEDLQpXiCOoxaWTiH6BbTSd1XNt/l9mKTKEKC2Y2wz/96qpHbx8eyMcgV4CzukKHCFZwBv8NGudLQ3LgDslftVmC/FyTuSA30/nJTs6eYA3bEzu9wONSEJVmyAPbOcjti9pH7qG8VX96mWBt3PpIf1FokmgQtVVug5wFvz+ejS9vSwcDlUR3QVR+ox0u4YSDoG2/BQA3xcI2s2lokwBww7XkE8Q0V1Q43K/2GTonOaHQ/jF56R6rte9ZXTMrE/PRgZOlpVm2cm27vbpU7Q9HW7Le0fPCJwG+xexyjLZzVOa/8sTf4gsBd4E/NwE5d9cP6XtvWAfcArlOIbcfVgNwITbck6D4P1btMRWjxN4ITR4CfA9i5AWuTsavu49B+Suz+dsLp8AJy7r1V2HS1uYA+9nJOOGuf22/gi0KV7cckXienyAIwaA+TKwEqmN/mQ21Cn48d+Y4Ow7LEUnXvQL5H1XJavDqOMoLe0xEwVu98uDQ1dcuEF2H+GWFq240Bvxi7OeIbBpewbq1scn7a5PqKs7g3zMtZTR0HHMML9Vx4QgIiwQWhh7bkPcMfj2XLjZrZ6vKVBidtclSBGgdioSTUnCHBHWOw0a+a3uUYSKNMnQtOIByiws4ZoOYzWkUEO6l3UHxny84Cm7t0iEuh1ZQXvqqPtz9N2sVmTf4/aGr9ZBFmOtl6bX5NMReGCxSoQmEX2E2Tumyamb617NYdKjNfiKNDIPqCODyCJJcGv6NvrlOeNaC23MUUDy4R2e2oDg6tW6dYeZmJP88K1wqkBT2//z2Zv6fky7OdK33OtcwfDyTFTwjeUD21y8JnWvpxYxdofdAwgXNQ+nbDhh2hf5zsc3QyNEKfj0/iZILzZO9L0TKNqS728dq9tA6oBPDrHFAkm+1RQavJa9uUKvs5PEGL8aiA6fCcvChmQE6ePg/0eTfOk5SRs4W4waWL1vRKAoBG3xEBCWIntudMGM20P5mGCPFl0hEAImjiiqn5eD72TiR94RlMg7av2NF3ynqb1CNRj25mxj51Ujc8JKPNmfTcARv20uv0ZhYxvxZMGx6AaPwWGRfsoDSm4TthldjpbSB7WK/1dOPJ+93w3G5lf3Wm8Xp1e2LtfjTcDZY65tTKv3UwvMCLb65w6g8pZsruDzrYu4O1/7qmBu+QniULtmjkHVr2bIdq3EOPXdKsGC7vDfb0awkoWlK4kPjaNnqFqXIdEUIuRnAcuMRqGdb1zw/BzRRemmfocDsm4jzH131mmvUyc95CR/z2ajT4sF546ax++pL1NSe+eoFuzq/bZFtUz+RzTqVEoofFtgUPbIKBxYb0Njm/quadSb/DshkyY1ADQP9XhA1vOxw2WSB165I6eLERVGgUS4Ea0Izc3OVZthsa6gfuiyLBKYIwJ8X/R+ciDLIEhJ4YWd+xZKfl8X5jONLr8XVil4mMlX3M5ccnMGbhlrMmJ6sjWJJ3OlYP/ze+SpYFr5eTBYMDQw3hMq77SHHk8ENLbMg7GmkrWLHCra783deWaB50FN0VfaeylGerm8M2wyfnglKG9QkbTqlF3KVwM7k50X6Glhoe41lK5hJJ0ukcJtNSLjrQqj1cws2gvY7Fd7rN0UxhSLc7XHefN6jjac77bE3lM3Rv2DIW3ASrBiYnzWS/o9arDr8NDjAB6Ahs8gd0kUX/rboir39tx0bvKjt/+8LL6B2GSbv0LbzzvtNrIuTdPD6zxVY+jlrgKf4s5sqnKNxrkATwsy1sqHHXTaJUAVy6moA9BiGJtiU8yX9FLkfU4ySIfoU6m+gHlDTYVbXg5OUg3C2f1LubY36mm0vkr3lqtWjnWmeFhq3ckhxpOdfC+Ra5N36fNChKgDkx9rdCYONcvGjbILL6OjgbwgH6ri4EBoLjAf4YwUU5eG4E81PXTB8yDP/IKgxar+VV52TrFNOfzNix8y5Ll3sXodhc0JKTAeQDg0FUyjYRCYvTP2SwSHjlaksD9elNLvwXItgMz6L3XoS1l+jttVb8iROKmp4wNCljo7mO30mIn5hkO0T9kI5Puh3DYtdr4vkHCwEZbUQCa/yZ1GlU2LK+dJ4A0nfT0aBR8VBwyuonwx4Nbev7LL+wv1vUxd2qNbBz/le1Du2BEYliBgJlYeywA/wf+pyrbJBFEyAwnsWBI7iU5c9AdHfboQHNDNwfgIEyesqTVmt4bybojqWQc0YSpk6zh8eZiWR0+QbUg6bggIyFKvP2+zh/EsRQ2kwtJaDzfpb3/NI0mZvQn2pjUvB/pkeP5JqkXuOQMyisC6IArYGE0Eq4vK9YUNkR+pspNVTPa3740kNa9YnF1oFl4wpdZCBO2/wegqo2jhQSGREBLZK+nxOQY1psLQEII7s89w4J8CIdl86iKzNUkdS40YLuUXIbYBB+vSi7sDPYRNivOmlRQFnI73+3O6u+99V0o3FsM/W0CQ10PlPQ5I21wEKqXeUI0O+SwcIS/uVE1DxQyxFb2e6he8nPixM/du+eywcgf9fEUR1cUaBsv4I+k8Rtr3YL7ymNqxfEeIX/MUkpeKqNinyl0D2zCocVqGQiAYJ6Cg2rPRO1QTh+aZySbsZz5bzsE1OgU/D9da+LlhCb60sceaUFkkhSP4jnJ4V33jqgufKFzEbplHYtczRLRzbXsflKbR4kP+t/YZlWxX/wYyk3ZJewz0gcXeX4v1ELjjaOtfCrGyGLkwcSZNcbiBbQNTtUu9TWMp9gI5BWW/4dcXDEKa2mjfPi9Ug25tWr9wWKNLl3FFFFb2WcMPUCM7GWqCF4gF/IQc5uIm1zU1X0i/e7AoDWTVfshkPG0MInkO5YivPeTb+c/wAlIK2JV1jLPf5/Kjb1mj1lcz2SkUrg4s1iuvKnTGOiI5d3fkZhSjpXXHW0FzUYFXVJbf698JSFPBeiQvK8eUdrNJ1zK/RhBpvlIbMUmQ3EvTaEFbePhJFLVm7aEcvSR54dFeiCS3Oqx0MeW9XaXK95H1MGvCr/LuaE6AYgC/Oe2mfU1XkulWzSchIMK2BfPZyGcqKfWFV4r6ngfpPXizeVnNnTKL/ka1W7NXpDuJ3qzpC3fAFzWaYdkp38akhinTKgfO6Gx7dJoAIgHFp9T9mZuhF97qcwrqtg7+8KmRawwg6ssPlksniMInPq4vCr4tif39vIHSivwY74rmf4eubvMK2YfmJBb56s/uaoke4k0WXk1WfSNzGyvs05SLcaOEOh6k6nv5sK13vIKyG/kM9CKuXTwHbzvPkEpAoQJCEG7IJHtNKvlhRDneSNKitbbYeQlzpb7QsAYNCTYcFe4se6/OUFI5OfShxQoVGi1vbJYvWCi6iE7G24glb5uUBHhU9zTV5kWoHGhSKF0ihs+JbPZ4mR0xrKk6RgLiTkkHwE0p9POjCA5YtzvYppyCX7P/unRV7pMBtnO6fHa/P6FXZCM2uq1kfA+wduAvdAjDCML7zMCCWmPo/tsr1JnON6iRN+l36JPCNRK7MmwGbhI2foxnTSxBpSgCSVpPIzEJbJNQkRkbhhx1bTPi2FNp1nUx6LY12Y/+A3HW/IXU7gjceVWG9dokH3MGwq5gpzRsgP6e5+/ZpwmnSrLpkVVfLMPiVrSBgpq7S1lu1nBHQzqeF7qLc3cuO++4h7z8Tiz33WYw24duA6LmLqvoqZ0DGp6EQI/GxrH4IdQIfWuzVXLfbqpAE7QJlb43yyy8ZTEPHP6YMxl7AT7edIhL/Rn6Qe+scAF705C4sK9gF2naEJdhq3F7mJLRnkyqHsE+JUB6maWCaXpI8qzIbtSEb7zo8YB6YO7E0V8zlx2IbaYXvwReeSdOvp+8PWQ96Uga4qYDb8VYUB+AQBDz8Mco12EUfm/ipeo7nJJOp2MZZzn3p6aLz8T1sd1W6lcMLLbxxeN1kv1kvS8d0RSzcbEI5HsW42PKMqRoi0wFW5bjk0X+WOGEvILbVnR/ym52LvJYTgJQoJp+W+12ANZpABFBzu4VSkNPCZRPAKr3ksk/xkTK5gmX2SJmc/Sxdd+J2ppW639g09gwRFqBK4XORGrEGyENRsydIJ4UivzE+9d5q6KrRggqpcGZhgLjU6Xyu0RWUp5DOCCDfXZ5S6TmCxo1zg8B1QkO4igSWYuaeXtffuvqMY11O8zW6MttMjBerRXYH3fdtz4MDyM6mgz47cWYk7rY49aB6E/wd05sG83s3i+NurtKZYu3fEncBWYyBNRKSOOhRpSHaCENbgPr2TKidTy5V4QW4JgG/uw2/y8OFChR7BdtYMWa5EMvsuBm/aEPObv2sH1oAklU5T/JA7XRKiI+o2Zo4up2qUNBIOd2oFwaLDurmd0q3not+JWTRtG3++3yHGRG6iJ2+rDirAxlxxSiTt/dSgA18CPYkmbWBFyqkhaofyWYqo37juvG4Ol5jAIs4a9VVSGeDx9L3Jcj+EHx3biXJcl156qpI6fEIFAl6+/l4PxfGYk1YNiWbLyg+HvEYZ9PvKOOSo1LidgsuN1T5JBaeh3QIoffxlhLW5e+FxrcawU9Th2bbMS+tiW7EnG5nIBCkbLqxbvfKX01GycuHAfJGLZxolP5CFL32GZo+4OLo0Hv36XcpJXDbBjO8pTBvysDo3mEBazN8V4CIyuqK6CqxmsSQD1l1WFkmrQrnD9U6hCjoVMHXf7F2ycycIjzCPreJ57X0A6OqCG2lKLzxqPflG/2xPsCFFzf+RV4mNkaVICH9/0RmzzNKus6z24kcp2XWe27pgJRV2Pvb3yuFiwWUxUNe5HxyL7Ym9lteW67SKXBbtIYRa6nlybm0MolHiXOrN6y9TngR3AWto459H9ecTmMsFwOANHEGWqQKH5Ue0jDFdb/HH7jZzTuE4Di/iAU9VbywMhYJIh03AMfFcjng4V+xSgQXu7M1CyfkQ65OWSLyEnMRQovwxs1aLqMcHez2Z646i6PBl5OMlHEf9Lc4+pPmtpaT0g4BuXifru9p6OmxykRFRXiFx5dFgM8zE+RD+QcAE/ChkqNOhD7WAjEl+WycB4PLvBcjJ+18QYlFyz9Wnl3r8YrQhYVQ9zkTlO8Z/X62gnAeBhIfdXpVVf2YKPKPi6nSIgDxv46k4LP4INr9OYaqQtxXUgfwBey4KY8xlPtSDzYgS73rr2+PNiQgesAE0mxPzu11md05alqYAo9m/nop2BT6FnJIWOvT/IPvS8Yi0mp5W0iKoEHz+tuFJYKj6+CTjYHnfEvU9u9pTlIDGuT4Wsp4bVVpTbrQ50vcWSWVG+NpaR4lX9qpKdGeWzePv8RzomCOgEDI8J/d7PjoHp1dj5k4D7Hu/ad60duYdY6Uw+KbXajtBvlSjeWLg5e3cUNbS+qSC7n1kfhvzlPNjMnkBNfkC2aIlquTxIpB6tgQB9nEmil3LeR0vabab97S7DYtUn17lKOQMVH1mO/9oOkdI70oaYdWeqFEJ9Lt9YwUAVMqZbmOnmYPKKy65OozJd5Q7CyfmcJFMc3JlakRn6EkKVnHypiAIZ6yd35gdaGarCnlCBgTf9zXCAz7aSk3maEBdhvGjmwtDsnGlqSZ36pwU5pDu77A7nj6beP36q31vkJ+SrBy/uvBiEZM/2P+MX2MvbdazAm3PKmLYxQyEwIKx1EaOi9SM0M+0beLe1WCGKqTyVOzFYcPhHgmnqXhjFuyJFTDjEjxqBsGPTXvSesubUC0KPSD8sw80mamcbk9T/+2DtnVO6So+LyMGWw5NeyU16VmSaO/8w91ubGBh13knBjGnKrqo16kI+tcd01xqmNX3tSRj4ad21vR+l8rgrs2dSCVINurCvkzbSaahOZ2dIcHqOaEAVqTovO512KU8WYhLX2JPsgOaNqP3qgqGwZCjOaJGPuMB4T7yFdusWbyJhYifbKWZhoqlOfpVtpUOWe+K+CJssUsW6LK1Uh+kG3HRtnVvf/oCKP3FTa4zGNALAl0ew6kBEPq1svhAhQk+MyDAmCW1u888+psb8qg+kqWoIAWMbEpQI2rdhaGvuUXLCnu49pFliEE50BFX6MsROSu3TAds0JwgvLcNFMt1VvUhr1skHwbf4kgbksnh5kefbq7K2m+FXfSrMh4tdnjB3l3WdyOn1cGIXhO2RgUdCP1nfuM8MHccZ5TuZAwx59rbT/AwLPo7XlDHR6Go2nxIVuOPgItokgyat+qwfGIHbrjzOdDHi2RxawWpjp6nzlBt8og7xKHtHvQtfpLC7gpHiYTVgARTiX+Ely63bCY6YxDVnbzU8tog8e7LrT9MTMOLgGTzSQD4bGGWnsvuI6wEgi2ckI/emLA5bfgQsfIkPBnIISClwjK4AsjFldpBbPG9MfNDwZiv4OXrnHLsipZI7mHi4ULnb5oYLt8W74254hydGiLBfESlteggQwH3bt9p539oi8VLXCSINEZu57QuoEII1h01lePz3ZBdQL7upApzC0k+zhH5KtvJDSwVDbVL7klHD30QUCqSX2WIzYtuDJgQ/7FxfHAtpnKI5ABx4JbIbDiAbzQ0mJq4ZHM1cQBwGUFsyBze+mymf5+kLRggwJm8n3nfU4h8VmPe5YEakAoht3Cp896xi8QCimiR1zapKtqw9L2PcqdMchWywbzgpZJIV6C+tFG6v6GCxSat1g37B9Mt0kEGrejtkR1tYHxsrx3rf1LNf2nv9rEM0O4RbewoJoJkDMLmXD1Kq2oYl2VKAZ5q0D1NOSbyY4G2sB4YqJnXdfxx5ltZItKZSt8XoDvALMP0rB+JRzG9Hwy//8m6CI3hN44jiM43IAhCTk8H3fb/iFN4IdHOhrEMpFX9MGeIO01mRVV0gyNYyC19wu1UlmC0jrP1Qz2Ow20GZAfnTJVSyIlwhHbY72KiZVApiWcOUk5MbMXmRclWT4XI/ZWeNAms5gkNa9ml9LVqeqahbCOLNNJWMT4qtA9k2JoAyO+Xz4WEdFpEi97rlUMH+i7eIDV949jap/SWR5h1TXUlsuYpJmr5vU4x+HDsLUTbH2m7+3P6tobxK8q7uFWsXINmxyWh6Me8oMfpLXRLyhT5W7Y1stJqHt4BQrbGMWU3ftgXQTAvx+taBk+qvdU5YI9+6Y6F+NIOl5MFnZ06YK9nSbec0uEC96qQrfBafGSb1gy8nU/wtc9HXA3I7rXR3MIYAF1tFSamCyvAQw9RKoxQQ7lzpJwz+LiArOmC7ytoZxYqWWPkGU+AxhYULAbjnGQCtf3bgsDYpnJx+UsEImQK5KbpAhrYi+PfW5dTT8Jts9CewBQT6mJx3NxtJSVrsqOyFaMaDR90cWZb0wePGo/qquMgaJhS6T/QzNcuubzN6tVwNfUUuTi+FTXgNw4Kw7ePH0smIkEYitNcnbLcgjMEH/cENmW9bLM6JYyeIkEGPQdPGP8v6Du7uJWyzOCCQs8wp7x7w6B7ovXwNnsJaEo5d0NEUJy63OPnCpi4fJB4oGeeSFFd5ABZ22u94uS/dvblEmVVDRpyxcLKxWc6JIoIK7gpQvdMdHe0CaVnB6q+2yo7kAjM7ZhIYfB5OguLsVrxPlL5FsYvPuO/KhB4TkstNsEoM+Z2ozALhPi3JIEOmfco5yT1qMosESKusrW0GXo/PG4VA9bNgnI5Tlwsr3yRe9RYWz55F2aKSg6IWecvzgcFdanECfgmFKIZneHQ1THPUpU8Q8fIpxmYX0P3Q+7edZn0XrQgn2hcOiP0ThK8cC8w1yDTpE0sCQeFHUc98AIvnPSjqt8abOmBFgkTL2DUxMGoeoTGx5e+Ta08z9e2uroWCeiweJeiWEAsB7qOAKIHajvMuEeSiWSFuMj6b/3e6OSD+wMyn0uzCuBwThEN/A2vWToA6+n7a5p17WYTcM8GKY7yrOK1R0c7nYP/cXcaZnylnk8nr+Adm4rD2lJED7chfnj4jpKDt//oq2WS5hxw3pwCaeKfhVFPPhVx+CzFLQBOf4irJmbLH5v2fXfXuPoJfDaXXyEcy8ZOalLqp6YHtUc+E2bnq17ZJ1HVE56XJzvtUPi5Wy4NM9rOUXC1ss+qJN0gv4gCXHWR9lxbW1SefxNYb92yDGYaoYGfivIecY5In94BvcFoyGPEivLDCLNHaRWI79nrEXftOJLExmMIwUx2pRumCxHojM6TkUycgYitb+jQr2ZEJGYv7kZZgtuH2l0fKN1DdfYAaowZ0HJrmxPZv2pFCrJt8CbaCYO6GcBdKaR+eO3JAxkD9TPFwnmWJexnc0uSdjtTH5pfaK89T4LRDB3akPKdGel96gUGh3/1uXkIMacF+GQq9ZAY1QsQ8NnemAfVZmXi5HZ4OldEdcUR8vhwQ2d9fdun4pYo6fkA+pZOHBaw9Peht/sqj7loRki4MpwKoGDTEvED50aT+U5VzSSkmQifH41FV+aoxot7Nm0YquybeK4+InmHb1tszd5DRovAzdJEsHFixcsfXK6aSbBwHZ2rxGiqDzNIh1qraUDNaj+60yD92SwwyzAaF6IMw/oWAwwOJGspdr0Ps4imRRYAAu0aiHn97LDXtWvR9iry+HEh4Ze1N6cXXFHOcZ84EKLR6n/QpHpPVUVeWTRpb2NtQh6NKli/YS7ubMs4kyrbcs45CC7HOKGgwqI64KBw/fZ8WzMq6Yz6ZRUPKDNAV4Q+X73m9a9db+QlLO72xAbtm5fzB+6bGeuyUYdcmQkcveiDUQFH1d1pFRX3WnJ2IPA0mZr3KejJdKvoshNRZJRUzUi0VZLAUv+c6sbjJgb13Rrfl3OA7q46W+uFPbkRBy514oGAnwBJgJvkBI+pnx9KXAypueYpfljVEyj9t7sgc6sNNno2j7ZuX+GzX2lZCvVQZUktTQ7h6LR+10vknms5N96EHKp5Cu2Qxmbo0yXq0P8WcyzmhUyadUpg4JjBHerx7iHRFGU538jSEqn1MS1anHoHLZogclHKkMeiVq7YVH1mb6gJjWsbRRCS9ek2d1XV2qC4bSHqfO2nCsQb7hhfiO08esRluo+k8CXyJEpheuc7IM9zEed0zM6T2/zgpflveSd+drQExOXrpAccYFW68L292sSHS3DkV+zR8UbUmiBqS3iR8JV614qb1K+QatGHXTU7RDUO8/D1k67bkAA+CBmtCDPKpsU16iNthmTHhwvsYksgBvVUNTug8HyF86ZLevfG78PNjo4mLoz+be88L/WG1IKR8HH9dxxD+lRyeJ/TuwUBzAWdTSO8tu4XVDeUDEGFjQfD/j6OdJYrNTJyZ2jAj01yEsTgh9sI6HSxx6B1rfVBMgPP+LfEKs0bqsTCGFCVNagqsK4dMm/G4VA4BEFGxPTv1Tr3vXN9X5YwQGTmTKXs6tgFYqKkLvE+zHfT4wbWDUchu2IqEUgssCupTwxbLiYp1N2bY5JTBM53QxsZLc1bmxyq0bcPj7LHp/8Z4u5EO7H3WcFH0oPp4X8k7ZGbAXO9pvwC5HRcV54t2okkLedBwLfqeifsYsY+eaDWmOqgkr98/AiqjLAJaPYaoDZn1qW1heuyduyvv57T42fpbK+JIKjm0bgalei8h0tZBM9s/T/RhoxYBRrS0ZMdPcIO3PL5iPQGcABouFhF9Bw4p98e4D+2MRo1DDJFRbc2s2GG1pVO8tLYaMX5My/lxWal4QzDMGm0kjEKtSQCKPxIYfyOnTKyk+2qzWJtjR9P7JSeI7bi+K9kihAU/nYZ9uf8QnJrS096jKXiwh72bIii2GLKKkC3iDHcxeajwxIjCGq418p1HCV80iOKTMkqiHrRrpNQ5IpB8YakYU7Muxr5x/e3CMeKdIxza8Ou0/4msQxTUiPUtir4+pgmwn0NdKElGppOSXg6yxEWBskaPRtmXL1Ae8tapIj0W1Rynfi5gN6YPfvI+pzTfvghMIxry/U7zsqvZ8PLdw2mhLtDIybM/tI/zrIsrYkHkbtYpmzgTMu1gjWOJxfSRTovZpiFCxCAxMYJst24ZzE7juVIJe5h/FU6vYmf/cAkmp+L/cdV1IBNNznZcEqusw7wAIDz0dAMzJsKtgFE3Lwbx065HgG+5NYF7BwGZyPXG6cbysFswG5E4dqSzd0rz7IIYC29FfJgcYNYJYdWf/Jl0Ybj98aam1BRuOrG8URGTqJTFv2DufPivxJayyMh+00KX/ZGLB/4lrx7QfXFPSBf78u/v02FEsRrQUELwOF0XM5yrcFcMJMHOYSrOCX2gHkS4PxzFlI4XN3GrGBHo10m/qLn9sxyX/lu4Ko+lKFBH2wrbOdVJvKg6OCpkDmNDRsw2SlxL7vuEbrd6uP9ZQ5aY1dB1mCMjihH6mLTuw00sLxugcqx0ZrZktZTxgG/mxR9FsZ0y51Wiq6+C1bTH+zZ8iH2xdLntnReGLIZOibl9MVYyr9rsHhejKZuNmgAKPZVEzQ3ilO0G3xYOIiF86iFUDW3WrHO2LUayFvxx3whGr2G64lB0BXH5meRVkFxAvaFMXoKiLHo6nEcSbiwqLNcC4tZ9pDYafrVoyLvVcRIrMZp+dCTlOgwlljH6QkPpUbjj3XOy0GLS+vCOSnmgeQV0LjOhxoCsfEzHuNHslvwG7vbmWjv9e5/R4/KWI/8pMbe4LMy1bCTu/cCavhgRrIlwZZ1/8zUrS8Np8oqyHgYH91dCKFg5+jwkxZa7W2JxShfskvcGed6tzmbzrFhyL+gVvP4vBU7+AZHMN8R5N9ZgkZgx4z3un6nj64dzoqM27yYrEyxMWPiDY8juGe6iE7cx3sSfS4XpZReonaDA5C8Ym5W610/wnUE0GvkOoV2ZyhOQdvzctLimV9wZxTQauWgzxzaFu9IJluBhmLOk+/x0MPXltJXJAnyhlGI9P/p8Z+lKy00zWB6CN13kCpirdEOrikAPrZgJkxf7JQ8/jJY/OYTrZdApQbvAWTMCbV1dqGWpC5Gyy06B0B+5rCbq0gPhh9uhYjj6nFFlJvu0CbV4VQPz0JNNjQ0ITPs3j5U0qCpQlLqE/aUcXptWNBgdQWT1qeSSWuBtr+2F21YhZotdVf+/ob+O/vZbM/+bZ4DLBU42IvbN1EFL3HsSYi8CXHkj9Kbjo+8DVFSI4w5sYj/xqiYCJhTarLk7KsBCwuHAVWF1kVgeMSTEYyN5+lyMQvsp6n/V6vpskHz2+o4zg1zcQEovS9vwXqL6KsiwWcgyMjHCTC24OSWe1S3t7qNMZQKnJM/LmdTUReeajtvYb9fHd9MIjksOVh6VtKq4PBXQrCE3Lac/vH2sb3FBpqFW9Z/A3lVoY+hAcJV4cl5NB+Iw/+mTlA4eI1hym+uE93j0eHYmWJEs8ZindrjmRxwV/x7bR3vMD4URbFcp2/zNApmiC/2YHrnjzoWda856PlgMjq5uj3IdMIyDva0+68rPBYoJVXAPwWb5MjvjYPIMN8geHbzo8sBxCVSr9kuBneFw+ktr04z25suMTgEtyDYa9He1IaBC5s/2ARW1RENWXbhcPO442RMB29AGhmNlKArSLQ48OsRZ9F6j/rkyJdLID6RZ46BEibYwkZRZ/LdcfSc9TxUlP9m8yt93Pmwj1RGCmBpas8FLpeSJEN67gUSruaXTQsZpMgn5l8uImxBpeJpQxh05Aa/19FT8UA8a3AM/h8k+ahKbg+HkdiEewEGxFKD9qPD9X4jm366m67UNRPvmWAt3zd3w2/v9Pq8ChGoCFZ1uy7lWk1AGUCM2KSroxn3jQzu9Yx1NFu8+10q5Rhp0YAsEYKipGxVkGqkF/Gxyo/Yf1aQArlPXlr3eoDLNh3UYq9FRz3PcBPrV6N8W9KEfAF92ELtvUKrRyLD1xqU4ZzncVrsxJAe/nWbB2PEAQjk4eI9rBia1vmAjKlh08Zbj/YDhyMaPm0xn/i8quTZ9LhOeEqYNvicnDWc0W+MOS5Ze/iO3V1En2qpJixX2lSLj0wAIg9LyEPHjlAVNd6mmLbneQu/RH7exW5URMb519TnX6WVB0VVVem/I/UyYKk3qoSFTmw4MSUN3IkLr/eMa5N7yMxi5iSUYedPmUncZiMMJKb9aBISF29D7ydL6p2JY48I8y+kbWhBCqlIIZQgIFKg9uIZbfUaXxQmitvnHaWNacglCpTfSRoyQ1zo6TlXwGLNrzFhl70VlTNFipRMeedBZn0XbVS+M3n7Iu82XDCfAMrT2dEfSQawwd0+DCkS1/FdGlXIbFsMDccPol+Skk6A2BD49qPAgx1+frVYnJI21YWZW/SXifowuVWRcVq8BPJUII9WSE+4BP4GBAEYeKsgjflY3NORgxdgd8o8AyAVxTU9sihTfMha5klFVCUUWSuz05+E61MT4q26NCGBDusvO+4cAu1U0/IrCy8exynmrlEhCaXfcj898nlgM+klqHDFRHhAxBb1y+K3MdGdLzQzgfKlTwJlXt8WTyQjipAXSyZXcI+PAMZFQTHPJseN3Dfz+Sp0Mx4to0+iM/PnP1dZ9UQtEzgkl2jnCCBJCk4AcVY31xhFRrXRN8FVQH1hyR1+fCnKlW3fTHMlq8RBxCofZ0hloMTCFc6qX0WdJkigGAvqZ6IGLIqRS5nyk70/nVotQcJiEXuVYckpaaSjxKglvOFtE782bgAEAda6gJH04En/GsyYG/7QYi4wzR0PSTY7blj/qRfVXhkKCVTBy8TXk/YXwna8JFtMIQDnTcZOL0AFjddfeG9qllFXFMtIo5lsURG0nnTU9CSL5eNx48d490o3SVI9gsgOiSlRh8LvSUxgOW4Lz6OLJh0ihyyg601m/nIz8f0834VFk2ROH8/3jIFjLLVXdX3UCNiw/EZd+2E1lm32+O0xU9/ZcKyBnS8AIgdCmFV5+WKH+5d93CMTSEypCqaXcfrkRcZUqXOWOZZ2ZbLDShSPPilTmLQRsAk45Ghn6tb5WnDCyWLxmhtRAHO51aaqWy3+wmjBu8q7cdyZOEbakHr34BxjE5Z1vEc0OTXoJUSJt9KkvDtJiDE3ksjnslFyzjjGTMQe9FXEpcVpfpqZE3kdCsGkcRk6kMVKbaF0zlFR2R8i7kZb7xH3E4aM43Bt1yl6cUzV5wyBGOxfCRxkH8doyoTGpakiEdsrOLA0j9Jf+pchjIleJcjAxpu7y9zqnfjYk5k/EuXmVvyo+TV7n9NacYyJ3Mcl5RC7kMtca9zVz7ZYQqSUJt9mHPG3WrN98qcIAEHVMM1XyKifGq0kFLyM5nXZdb8hFsxLZ3X89J/GDXOazmJ3WrAFH1BtiYnrkGK/ydS8u/B9Cn4ySFbwYWlvFRGoF1FG3xUW+MP/qFudDmBGl5kXJj5XcPWD1IMh3Pl2cFOGIUqttuZ1h/7qcMhhDHilOaqQWIFjqMSsyg1FUDUq5wLYQOlY1QpZ2Exmz7YK+qpdYlAabcMoECF+Vi3zz8XfoOh67gBNAEL3xH6Te4yAzHwruVbInNGQc8uU7dLIjkEIYY/aE8Vo/5IFnsTQovZ/5IAE7aD9wCOKJNj0l20VdbtceetRCGK7kbfTlO/95cW7bKzSET+kCChWK2mVorxXN6ASnQuYqJqYaRKT+a+K2mlCkYKa7a0xzB6BBBz1JSqJVWxSSo9mRbvm49y3FIxciPjXRJvrxFwT0szRvTHzy4vvcF/ul9FO/wmdAV/HNubBPPcuNUdbXhlEfLlyGQPZ07TeSKb+kaCw0/nbvwqtr3PozuD4Nmr6/RjlBNskAMtRmHtyeoQ/zhTL+UTMb86A4nIVzQKeyubwDZQsrLccUvQWHdSSNy9RTJz9cPKeq6cxE1dnrbQGrdGoZ/EKf+CvtpTBSFV/ezk2LPhyC7+RiwzYhiPMr25Oz5p0bY4swgYsTR35StKyjgmTerd5jikckR8KRoIWUqja7s04tZnY2UxueOiL+G/yKEEF0emB3wlodpI4387CYkF7oHAUCCzFA6TZFzZzSdvrxm8mARbdGtHPBKOsvz73VlQmJiCj9QJF6wMR9UOK0vF23LiOpQdP5XaM0i7WOPpgwRPQLQLcG+7MiUiBa0EFjpOxLNR5f5U+P5Kj5n0xJS6cuz7vomP9Yj64TUkySlFTJwbp+EJNKJqGhrxXkU5DZ6TnUW7vGDE+HumtMakxka8dBKg2mPPAhzllVkPi+kWjdTZtnq0ylN4WbZxRUffrhNw8E75LgDiPlSzbg3aNw3Q3eBYsRofZv8WM+jqSBRGfwsa1XFUyHAs13E6/SeJCyh/p3Gk7zdIeWZLbxq1+3uduYW87cUDcVi7xPEGix8SUWZIrJ7FZnQGNoNkWoXZfRry4O3jhV6jExjiYn6QMHhw9eyzejvG6/LexsdLz5sOapr8g8cp/tBw/9d0nUwqeGvJDL+1fPg9W2ep6BVmLW+IXWaCkR8kOAlz2tL5MvGGM3JCu2zp3qWz73IDVZQDaGaxuhZXbSGVvJZhvjI+2wB6dSt5sBsOdUGS4Ji7PuN/a2zc3VBEF/JJa8ZRptB8Pn0AO2H3vcsUXGuIUGxwLetXosG2MZVhU/iOh6wRmA++Gn5HJhEEyFxik1RoSa+/4HrdImhpPBASIuMBc3/W0p3XOhqmS2WEXS84m2WfCph0WsLwGh89ZmFRL1NjRDeydaqbjFaByrY4CWDVJzHmlFL+W4qcInEuBNDkPMoZA2krDjwtfT6QTvDI/iQwCvvUkhgDmPu0zMRmV1awrKaqoQLfmMTG48uf2g7NnXox4MT2x9Cblvfr/nfV6ojXJBkrmE/hqHvONFTUSYmdmImAijv+1iuhSZj6jwNAcNz1ah9uWTgNYeOsA9ZD3d8TrpwUD5YZogrM2nKPA0gnvIUc3YDBtJBOIoB1Kq8cdsyd1EEXi1kdt0u1mDM2GajPNtpqjQ629gmJBk9bLaUKIecIJ7sI9qFSi/9klUT8rR2UGMiNQ0IRHxhtXgj3cN6H3tXD+f9+Dw5RwSC3O1zsfP+R9qiSoVPBlhQedpkuASn1vzbwZUFpJO9CJMqGcY3elqah3f/I2qcizeCrVkg5Nc4vqPGFLAF5QTH8+BwSYYOSiwmy27m47PIwKV3wZSiXxspQcvk343T+muynPWSKHFCDVeuzUIohfNOaxqMF3BBG56SXUByWXvm7ASKE+EM2srXOnIrhPbRz/2/iat9ObsXJMjRl55txCUDhIU8Lg7ReUuqy6Hd7o6HsSYN8lkBTzHHQWLMecyF91fBEetJ936MkMLJViot+B6ILhxp/nogpOJu7vLwgZCs13wCvDTXQwGhpaVu97fAOsShjuJrYXM5V4IDM4KkcqcasUVZ+RdaKofGLKCyFJLM0T/n7syrSrE5UCAqOXjW+WNJS/pzN2x7gUzltrSTJXGYTzuDTjBN8U9v0aWFHANMb6Mz+HgF0uRWTOFfFlQBiRHhSvIBAS2iwSuZY2TiDMjysFZMqytokOsm4hcA+dSzev41GGhKryU7zCGRIAB7m9s49r0kmi/pDBnGxWBV0gb8vehW+8etQ7pgGJYoE++wHxgTOX3fTwIMNJd3i2O4vhuRJXk5JfAAHfIEqa4ghnzMIMCo/rZHfc88ZqJs7hwtmIwYAerPpgOAOsUYnRTXhroGC9OJxA+WyfdakGd0dwO03ajr8KsBOfQX2si8A9rZBVPk5WLJ+/V9WDB5iqOwY1IErTCxGCX3E9McSe52/BOotkoMo/qIpQHIRcul3ayxyS1QfA81t30MQli4x3vZlaWmJLzOVUao60T66iQX/mqZaFn3HpqK3Kv4RbpFj1kg6M2aUS7IYO0uEGciEhgAIfSMj61cOKOyIQZHnp6/yui2NU6FZ1hDIS5aHFIIT+qrHT+UneH1FBVYLq5hHlT0CKbhdzBFIs64vREcyxvLLjXxfqeSK57K/nRyKp3tIf0IBobjDfEXwSK/htxFL4rdtriGSYllHu4wrVUHzd4DhnYfORPMsugnMliDwrrxNtQxLG5k8JHYpl4LCHMrWI45aUw65rPrs2JHoZQpwgWJ0B1COW0+AKvaranL0u4i5Ua1em/LgSM0PlskDb176/5wcjN3+X3wilRdlArxsNQktSzgGKOc12KeUTzG2Z2Xlk3eKnAQF61Ln80Qn45COkutfetcoUVpELOKdLCTszDuTdy9xz/Y+Rtdj+NGbuvkiypuXEKVH4UtQmcw/J+pPWwqXx85/8XyHoDFMy1ed3ZaXOfQPWoLq3XoaiB+HIJRFVxaFJ4WCxTDW1OuSfPPaufHClE7Gj5bfGECU7oQK7/lbzclN8jwfgcLTNjVWa+XsrQQb+qf7IxhidrctIHuOEICTRWJEVXRv8cnnUDOM0hUIx9cO5zVPJF/oDhQpRRfewinxqLJ5treFxjJOTwKfunw3skwmWJVRz5UgEQ4YkB9SfqsJ+po6mfb5Do/OFNqdCBwyVGAcoqfWrM3tQHwvW3vX/Ejg0as0UskHyyE3SFHRcrm3ZfPi77WD0+AOHDjv45qQe7RVE+Ntf2lu+DRClQYQZgCNZG6km/dCvFWfPxvyP1e1SDCw0mhzYGtwgRVMh71TVbJXMLbFtt8pIJptXrLXvAW15Sfb32cDtD5Olx7WeBzy3dl9mggemJuOum9dLdgE8bgUHz7XeYqnAq6F5swIuMaTAMXpH040jROfc1a8pH2OMj7GI89oc2xH2b8XxYkFC/OouuEax6Rp8E+rAOye6QHeFLGkzPvBxgaF2oimVYyr57N49JSvJcofXLYF8XN50A8Ne3AKvnmjlvketIHPC3NB49kidiWDwVebWsCNxNAFLqr/f6cU1n2VWANAAqulK1TeD5YX3rDD984p0i5KlDiXA4aJKvjtyv4LhKLqCmBx7TZCEAcrAkYBsa5A4npC92y2y9n+bv8VyKdUh+1dc5opK/CzrfMG7rdMcQn0TXZkKyeYk0DBsqgM23uHL9ie64NTf0kplj686HLzdPGxvOJxRK9WfjUFYU9A7AvDBRnLL4UzaD6CZx8ovCd/sq11MLoUAeYnoQLFraBWC2jvXbnP5T3DZ/qlT1d4mVZ/9XXYc5M/lj3Z8WSea6FH/3p02tnqqBagA/e/fRVOIhuh4Xnsu2w1OpZs0iYPzrsV0aFYPa+mBSwUNdpIGC/hit8IHPuPw9DSZAmkJHMgctibTuniiK4hwawVEHPUj2GdEvaBiT7kQo7huCEHzdDzsXCksJZTX0BM2See6xA+kFzp7IYlTeSolGcosymGctbI+0dZEpELy0/Fzqn0zJeUEJGrYen7rGISHDJPmrfmZPkVyVgvlWrlhLZxcvMXZUYm45lD9cNK9p/yKwonaXFzEkZqPzgBHX3Y84jax9LDRGkgQdSyENVPoSriMPuCQXxr8aXp7hN4U6QfYhMoDy63BwdhM52y43X6mVMUNg84OUjEhuEbq6egClWREcOJlY9NNUnrWrrJwIZQXzyACTl4sXBIA4nKhBJQs6PpsNqidEwzUU3KoBkqAEnsT7ZCMJS5PH8CO6rChkNbs3djm8Fy9ws3kqJbZwvQM/8RUP1sUGhFin+hMEVPgJZbnQBIPyM9IGp83WqcTxBX0msN45AtN2bRZ/6eBGh23ORyDIwty+V025Ch4Xu4LF/tCP0KIJvnjQq6SebIhBl+1v/DhSbGMaN9bV0FwXfCK6zk0SmRWDijqOVla412ddlfEWW36XLBOUpTbWM/spvmiOHDljxph10TaTMcq/lwrbgcoqnYlPfwM+hUxBk1PEn135Zbm5rUiG8HSm8kQ4WnGGp2YWTtgpc9nWqJv3Az1cT+Y79WJbeX3TRW4ES2Gkr39QtPkFcYIyxvOAaaIoT6JtxE7Zl6rATSU08aRtJSestOEkKgFwOZtrt+WA3l0x82Bw/ra0+IPFAFvhMNNS10VRt/dmvrlOQq4ix7wyeWFlxDYYrZ/wZS69B6RKPXDYt9gg2hfCkz8QVsVOkAUjZhBOJWdG9SymD49ZyeIJMSwFfAxz5dGWi6oHl3s4m8+63tbglO4Pvmq3LbxpsdO3cpzPxwSD/PY/AYj9wq2+rK4hP11jR1ovhxDb55hnuuXosixJ/8WQRJ+FcYXh/ommu/CT7Osh5rMpl7GyeN8wP9Z12ICUqytRMWRhA+3aREXAdLzfvfCJXK7Iivs6T/62dxHKIzbBgBfnASKijM0iIhbddQopVU5niYckvSIzBhd9CvXbeuSxOeZNQY6Nvcj9Jt+nwSsi4gjQuTh8QzU6AvjCiW6SWOPGiWLtqeTllcuL4dS09t3gyiVmZjBpPhwigi5OlRgmZmfeAbuHH/ipeHPXg3W5TQymZa24Y6BLQMQSPf4F8Dw64nHlD/D0ASxTN2Rr45sJARrTWc4Ho+bvUZLAUfZWa3IsDc0VUJtP/sN15r/VuKodS4zrAUWIMrTLPIbv9zTgGA92Oa82iDRtBnUN2EwaKLU/L9lpVWE8Q8nt6TN70j7TzpL2bOhsU4kFV803ebVYvGwgOJzNpxTkZimne0MhDPOJc3/JILfmx3syqMxO7Y16AkDZ3+2kMj7bf/ZpN0/WVQGxMUbbbnalgSoUHbZ5cl+WgWwguLLYo11zFcxEENXSTK9wydfROgM+RVZvpq2PeSH1+NcD66fgY34I/EIuS26dkRSBu6A22+RLBY9utIzrbmtXdTPk0zGFhPnTtoO/BAkwP19Icp9j2aTH9OhU+9+/SUmAWbvY0pausj0j6G7KDd44VK3xwbYC/chJFkVfDfGPSVie1uyts1SGZ8skEI+MEDfwL81oP1IDJePQuDN5XmykR4+DHYR9oTXpYiSwHpi0y30riZL5RialsxLRm0FaJiahd3WDP+et3YSOBxlz5hXUGHkIO96/o1ANI+e7Dy6Ts8Ygz3K9Q2T+C8NPREwgWYOCAGhWwcsyJslpO8AFqmcM5nqVBLMt5iKHKycw7/YgdDT1K0ejDe8ryDNE3Q19mcKUWrStUHZIYFBIVHWpdKhyB1mHWQDa4dQK7UMce6CVzHpcYkAOBuIJFydfobFbxkbhU7pGSAB9N/NW1SNU3e7T/yBAxm8QXziyIb7ZtqBfo3L0nk5/kZdBvMx5h4uAAHiB5gsPKN/xOfHrRL+yGeq1asJ1JqPSEyTaLgq8/fU4bva3iKU+YhMPphJWnvHEtycIokuvE+5RGwLBFxr/jWyMYOwuLc2gq9CBuATp2d+WxU1LqxMo8Aqu6L3Edaoo3+bg6J+cBBPOlmsWJargesqMPgW4IDv+acoJgrqOHdsz9ftl770HCMV5bnRMPeNV/6SUaFH/GDORPVXRtPyaRoQmTl75tutEriI2iz832puZsr1ssESdMnW5cXQzVDYY8DJ2s3/G+Is+15zYvNDNQ7dN75bxLpKEyKVvQoLsTR0/S8PoN/5Oro0BwG9flkxcjgHir1lMISDq9UPLQcb0nkBj7TEC1d8gqPmzlPK6GpHIHOhJ4XiKY1k7aaCzRmHOlrV6iF0bgtW4QQ3qedJdROGTWKI7kufjkm6PKM55wgKHxmnZzC0EcN135l7JJElYPlasDX33Xw884U/x2iDZnD3IqYsnyd1zgR7kubbgXznh70zhcjDoFAJojzDLy/+h/bk/qO2faSugpS6T23GEHxvKHubSLkeYfePkuCiQ9n7Re+ZUz7hSm7yOFp72dQUiWa5EXv7PBMJ6L/a4PjxW27OLGsjIgB84YHH1nFHsD/TLqBORVXLNJxQgKphKFt/Pdu8xqb3rEe3MlrnrQ+a7z1v1AEwwUS0SKwJhMSS98ZdzLkZEp64iOvNdTP2LmIEnkYT73eT96lopYYMyrZT/tC/XS0sf0ItmPvhv4JBN4Yq3h76lto6Z2dq7mm/fQrpyOJZvLBvcguhUJT4Cy61RkGJFWGmDBmTK4rn6fnOw3As3ToYB8K8EbppZYFVPzlQNyTJ0DR1w4HHpDrbv+Z5iQNAq0w+bQMKrzppT8ocx0cbXS9vfIKdf8n+4ETHF9Cu2Zs40liTTniLfcHy/drcbLFMnC7pdtNH+pr3EL5pCNDEbqnX/Nvbz6/fSbr3rDifYSqrorlqKTyaQzNa73+TVm5LiXnb3S+wsT6vEMatyF0sewWcutzeFAwqsqzgrs991rnkrbmo86vlPwCbg6P7V1jNIEltPvkOgO7WiHAatZjto3cBWI1QOH9LD0dr1I/MBXG5UksuQDtGBEwOwZ6BtHcIGcrxY8CxWNwt0kIeoQFLQvq3OYe/soKmWD23YY47EitISBH6ds6xtnD8w/pLVHWulH8MUsYdUCp23ROkQF0SQill3ORJD5yVgtz0S4XBo+ixyIZj6a7G69BPuma85r/XccMH0p0R37vBbJYqZt6N/H8Yb+EYlC1OwFTJotzP3vkX2auorApTdLeZkgR5UW5SR7kuax8oHwQTkXw3SOFLBwEDMxoXoytg7rGauKHv6InvHCUykgm9PaKATe8etRIGEEuAC8zcy3Jkr3BSKhgcEBKuvZOTNwKtH9exP7DzxrXa5n2f598gZPnAgIcBDYxY0mDRjYw5eiyJLhqs6V03wHah/6+1gUUwnFNAHJ0A2xvl2UUBl/n5ukIbr7OJnjns4UA1vy1YWNVLtDER2nmsZ+vbIihFtkAJtSP0/Wf0kfTf8Q2FOtJ1GXb9YV/3hUf4gHllK53O5TdjUk+2u4Q2WWls3NkCyRieAGahyCGrNmWo2iB4qT1BajWLvvz4SK0IvPPcztCR3MtBi5B0F7eTX1F5kAXLwwLuyMYCvf1plLd1RwLTFBQerAoN5IRHwT2Vyopop6Hi24S9x7IrgaFaDjc3uN4dsiTX73LGdwE13PFiSlAEmwEtpGAhVyQ1PrJbXcbUQJe1O1g4B+qXvI4sIW6iuqhLKcqldKWk9fs9xSVfscN+HUXFqvg7R/KIQ3JjcPaarznVh6rPLGx8PffGPnOQeEV3yfhn+4jq6kwNNupB85IXnMxdLUM+VVc7aQoysvqaRXDQWVqIakqB8s8vgBSsVY0p8mj47cLgeSR13rephs8OOCXJeGjR72ZOVlBdDWIIR5OOHM5rDNfjKCgBV3mB7MuhgcbI4ozFFslQzO8Z5JB/gv9MykMrKepV/KzNMKWoTj3+m0lO8mRr3mWDzECQNqlkoKqLO9RmO4NnP4+fBRI2b4OL9zlkjAv7HLvDB6pjlKiYBnKLI+qJWGHRgho+wcknWCCARLmQ8Og5z6RmcRLl6fZdrjtuw/x+suhpL2W7TMA7Ocpj4uwR36VI5edU6L5kO8VSjajKun6MVz88FwqyunE2ZhR06s8hb9sB7XNzIGdgmmlBa301XaDeCyE7oGoDQrd34Rbksve/UEz6gN2JcwHCnacy3UeInkNlHCM3XY/eQhmOMCvUO4OinAHlABYywMzb9gSNSUhcoYlq5zO0MCcgZSTT4a6i8OpHgTOGLBDluW9Kp32GbH5qP2HAh/tBJS94diBTmvCkHNcCzpjTJyEd2mfGqk/ZxyfPyTKbLNtckJfld6QcDglY7ma64+oTv8mRGm2XBDzvVrTd1QNAPMV7hK1iRSMgjVt09FXMnmTy/KRapsiHcjX22jAfPk3N/MQqfCsEu4l+qmIllala60YrD7xV5qVjuliT+70tk7gLI5OZWuZd+SG3NfPcGyZ16dNvl7KebcQ5uAu1fOTHMUUXZCD3WJD7TAzLxVh+WBcnpFh8zgJS3IrpFSwWWFYosVNnaVYdrOWFBgN/d9tS/dH+iEHoqMxjasrDdjZRctGF1jktPOyWBu/Ny55QpSD/mqBqT7mlNtV7fYnJUeCO+PNdXf6r/9MdC9bGgpLNNrRLE4pMhgnpQ9Wd3Hilo+XNnurztTvbUWee7Hb3PSMPYx521aM3IfXm2vAamMv5zFnF5eXJ/+gRQU6B5boM2H0yGRlzgbh/PWnS6jO9X6S16+1Bi4R5yXOp0u3tXdSQhShdaI4OZn5HhXE1lncuWQrayB+U0hkyITvKkXG4E8OgM5+4lQaFo2NUxMu46HevgWRufNZHt+iKnIFE1t7/ZYYdiVaZX/+2Sjm+MdIUj4TTFDcHM+EAVuM5TwKvVZLqk5h3qYcOHgPk83LCqsRWhitrDqUJEm380qxCRhj9O6I/WmZRCLAsmN2uxWGzABo/oHDQAqxv+cd3EB8dOsZ5fEo8fkh1a11K3SgWxxE2WSeKqFLeiM98f+DncsR0rHH53MLbpEEMwo6A8GXeEzZjrq5HO+Dw8fqWMkLgxjK5tmseQuJRd4Ql5iGKJkn0UKh1/TCrocdbxnYPN0F35JddDfzAXgY3uWh2aVW+CEuloOtaAhZc5TN4JtqxG75esBXcJVTfQ0evLTLOnRgIuTvd3PuAA28s8Xn78+7wBPYNFK+TZn7nlF7mBoM0VflrfrY5JrSxvMbTULSoAAvXR8BGvBGtPWi73EMv48vfKgbEqAkXQIg134GbwxtNL4yJ1B+uzJtumjtguQUZftIJ2pKez/J3lObP2mNpzoPU5AIgfdoKyqCVihuSOfRllGL4KJmY22cVrAPkTDHi2q7yMn2twejjYPn8qRzB1AuHXIkbpc3jofu2EcznWR+q6lIT5rMarK7keyQCGox6prGPeUVCMLWVMhezNmIWLPyqOT5uGfxbZLBjVw02Vc1cIP5yusHuP9YAhroYUrH5f52ZA20EAmn6nOiR9kGYw6XG25ZSs/5gpF+yfohpS5JzogeNztur5BZIjV5FOMLPYbDkycqFKOAUbJgwnAhP31Mv6piKT+VxxXp5SzDWHhYODqOuObt9vmyGvCpnShGaMXHaFC2R2vXmR34+6MUCBPMHMqwG/vpwAlYaaKcxnpwp0+/t2qOEPpHPVYgKjMG7lUwZE4Dj7GcieQGr/4UA1ik1XLIfuLv0cnnyn4E41GWjwM8ltgoo3gD2C3gCqUapa4v4xFrJ+07vuCzBoeMd1Ppl2kmhix0gaAXg8phS/qgIM0VFZMhGGBsVQJVVsS4wO/lim9Po5WFqgseuj3EyqlEDAtT4opUiMF6w2poQX30uSeGNFgLqsmRRy2h9zlXTY2/rMWzY0NBkmGpeV3xvGbLvIdk4uW9+YdIK4PvaDdvkfsGFJZKNuC+e1ic+NSxg7A6cJRZC7BSMWfqwlthGVoDR3V09nK/hG8wmhvRLxeHZe2giFpsuS7OJyYxC58Wr6ymVlSPbZT+7EyqdWjbub0G6f+j95eHy2FZH3+MvlPcpYObJFaGiYlx0dmxMVNi27iMyBW1bdoPG5gznQn/6F1318Pa/ja+USGBYuE7eoWcCZxHzOM8rplsYsUjuxVPevQe0yi63fgown6tuZUTdEhJ2Ac9KPlXi3tIK1GmNoh4JHJsmGPAEQJykvRNDl2wsidhL2CrVLvf7olVyWsUP945LBTxWtoYY8qTJZJ+sTKxm9+JROhWLE54ORNAealgKB5UHCx+Xi7S5pGeb6wEUIRz5EsMJ2OTp42MuJRa8w7aVzyw1lszTlDaYx3ZU+EJRKhlccMVkB1RocPfEIPy0TAWRLeRKKj6G2uNe85ttaQ31CfTMHJE6VaGivyj6SUSPD7wHyINZMyvJfS+zVfra49yaYegRsF5KzdKOJzLnugAhf+0vZyzTY+rpGVIXqsbP+Ja5le7NySSKSAPYtbxJuozmPXcog05cijnSOT9LuztUKScIJiABbt+fMGNkcJ32qVUakWsCJ9jlS7TaTYbE85uPUOtOKeFAzeeQc+QOvm2YepDdeOpDOerb9vIBefA+d4DVEuLdW5+DGS+VHlQoPC5OppROrQMACWAVCrrhA3BdqCF5jHQcORKyAdmyUc4gDErB/5GK1Q46UVZIPpijk3GS9zULZexOaMZoOa6wHfjZPscc8rRGBPm8dyQHwTUVRuwfn3zpD5QXXSazVUH2Z1jRWyDsDNM81L1IGXluVhgB6VDkSKj6e5nHEhXBEqdXIPlzAALs8EfvQJRL2i8fEcZeaCKVxOP1iPYFR6wgdoMJqFwNID6RtIFXY8Qx0Swy6YZILWMP8BeDpPk1EaJEPwNgfpAq7LOJ8hglPOTmzDaGR4C5jrGga6ExNx+hWWG5BBD9AGFcImKN+xXgdofoHcO65mZOC7y3QM9SVSBWXLhVLzAZunmEZKQp0yV7ciFwvw2pIdAhPd0kHSdKt6CWo+fiis6L5gQB204Ipe1JzuCb90bz9BkOIO17ajFMD8PMgsje72dwN4M9SZzM+LneMc2YzUmZnloKdCeFUaQZCI8xDT7HOL+N8L56PjF98VhSwgpm8kF1O3TWT1WG8vkDDQL1NSmntPw6rP9jqKRNs0A3WDuEC+NfTXwNxaQB9T2J5tFB1s/6Q0KGSzI3CYZfK8UzPHMQE8YQE9W1y6xnzXlqBXT3KWWYxt32aUtPsym/ZAu77rLdvsBY4cXyx41ffKzEhNuVn05rs7XxLjn5uJzpr2CyGIBEs1m0001rLyRWUJUaEJ63ms7XIBfjpRDkFJSi+0demsQB1BkbdKnTTZ43GFdwhlPPBGOdEjUepWuciNNE3Qr780nkoOGn31QSxOsBy9XgEtT0wVRlbTJRgeSMY7ZlRpT8SPOwLpHdokxK6DgyhF3MThmb6OC/UwSNuIR7CbXbVYlxBp8+4HeEspc59pk9eFwE/T+K43K96NLAw9SKWP8IU2Tq0tavRnp/iHB5FML2Qjy6q6kiSSf4Bngzxnl00jDxARXRf3/Z/m7NnT9NPElxmw31jfATt621YmufwMyicdCPKdbPeIxPBzJ1X0IFL3XeymDoeu76KYDi+N34Bt9Em/2rf9G4crVWaJ0rA+2VcN+/p8DTaxxXRcm1VgLmmzRLTxORZ/+RD4VvhLTKyUU4W4vfclX8pG5aKWDS4S5OIgpN4S8F1Fj5Gojq7Yh76bDO5iNhmb0OqyoY4puf/uHekKLPUrDTscV3yZJr5zWflK0zShSt9DdEAubKjR64a/pqAO9obLZdQDFPHUeMpx3cLXT+HJ7o4aa232OyQh7TT0voB3QVc+oUzQMVSAJxmhpguNz4j9Xtu4FeJwY7OHjrrIzfj0FtOm4JEjS69eEYpfg8g5T125sRs+xP5DQ5gYpUcejSIAkReh6O2e5wGOBHBeFU8qSleI0cS/zLudvRxVvmKXWkmmjAiHW+UTVkaqwoLdru56OeZNUHQORGuNIEYzS3vUJ7q8vgf6NCP+et3TABteozDjNlynieMIv9ZH8zLRL6rezZjLWkvFTAQRaFet7gErhRqzkcu/nDdBzth5BIZgEPmiPZ71qoxkD6I8Mwj2CXkisazkm/k7GXhB6VzfkGJZImH3678rupK4nwnsi9Vd+RJYpn30ohf6Uo2lyMIHU0gdHnpoU3yPnWd5S5g7WzQjdoR7QQ/UlWVW8z7IJAeVThOtoxAkpn0R9BC+nxDVdJq9ysto3N5y3OAopErnPRxiK/ru7BTklwJkC3HggBxIkIewOp8M2+hVoyf4HQc9hIPjRscwSAJGXar3zq2r+DDCOGPqzoSOUIDRbgoGQ6FtGVwnGs3PV4wP3J47UXDMBV89OwcevCkv9DN6Ika8SnxujwYPDJv0CZi9S0gQ4XXLfidivVlRxkz8VwtCd2+2hRXImHXHyqhwlkemCyYEdlpXv3bmIigHNbBIfhaKzH2w5ybterFirRjm83X3v0ntPGMVHduKc2fz54vm33tfg6v6Kms9B6+Wt5THw5MuB5Exy9k/FfS3yTd/xw/VIT3/BC6APxT5ZL3AhFELaZOxJwm8sq8a0APjmJX4qSsKpdbAroOoaM5HREeza0AF4BqIKxeiYRnsCl78crb1siagQPcUtM/mfUIxo1VEMUEksvkKgjdEgmoh3d8vaJp9BxHQVxLsurvh3IssQRaf0VhodiXF4l25khZH3CcUkwQMA9XHeEyTq+XcvQaKXMGuLbK+W3CNzLzY7l4jqB/s000zO5HvyuCmMtif+/a/7I0dDz8ehaFGyefVclfGNL6V6M3c9IO9PhdL9d992tEG0umqfjBJugS27pA0hkilvmQRgtn1sagJtQL0okbBL5kaa2lhqfrx5vZiPQrKH/Fx4/1YEVib0ROKt7eaiDqZjde3wgB9WgBX+OX4uJyxTTJiZL8n/vya8ErezlDgsEdOwGTzH6+l1W4rK4FsBWOHDKR5H6l+7O9WuyXELRcsF2tJHGUu4fIp8g0eooidFjSGWRfhwNduPPzmOulgAR1t/Ejgwm9SsMQ+JPvEpWjJA58Y9SfVd80kRUpvf4Vi5alZ6cnRj21vNZB//BW2hZdA2BfJoy79mk/Kd7RYmbfhS61JwvdRrTWwUDWxYPznDWoCtdbQVJdRxfvFNEjfgzzOubmP/Wip1d6f1sSESnd0/z4RAroazdDTZwUupw83dNpQo61EVGlq+v+wgyoD72gMBGH5KaHFsREXtqfgCgbZd7jkGr6BCiRRKLj0vP4unFECSgJERQ19xuIYKbgnx0T79xnzN2R02+mFEPWZh0GrXeywOVQgUDXOON/FjnIDX3Bqj0Df3ht5NM0JXK+Iuh4TBi40dEpH0ssRgA4w0Z2ukipN47XUqwbxTSEf/rvoO9OPrq1Gg2+LR0LnZ7h09BHFKhtnmag+/ox7iyimXMPNtLtxsv5msEBFNaUAZRrajIymm8ktAFrcxgvPxe5VNO3TjVEISHcMI0+EvyTKNhdIdi5qhUzKMruTPFRKYnW85VMobx+fzfeBC35458l0vGbA0LHs+pEaniOu8maImfLtUPqWKTnGPhlX2rtMvCSQU4YXkmpyRuMYc48Rt/5kLtuTMPsjURNyPeuo1KDgZnI80ufTxnlIqBLpn0sP5wCE2yAiavqSjVq/NwUJUNRdV5/WhWM+pW1CvBEtguyfZpeuMqP94KkI7iFl7bq4wn3PU5LirOVj1pM+oyubAc5CWyTDThJcVtpLrib9ebSqpZlQnjSzZZ8/xVxWej4IbotS6Eh3a91lqqxMV0PBFv/f5FGTEu6Mj9GkwaXXWwXSy+YFssHWbwmyzwThtNLz+sVi3ZrpIVwcXd4eB6RbOGamGhWRfrx6VJec9wDyQH12DSsBUvZmAHnsRFOGdaMJpFZL1XBlJQbAE7Seeg5Ln/Gyg096PuOBcgEzBW5po6NTSa6ebJ3VjGXwdyrmOlasEL4auUE21xl15obqPBPws6txmSYVqC3SYQ/YV1ZNf/jCeg7XSIiNtNKlmWpp9K4MGIeS29WbLxG8q2m0w71C7MD6PEVMeDTOjpSbCv/kiaxae3smahahD9zYXNYjY2hwcb8tuEGHcVcumBPcowYFcEUyJ4wjQX1DMJ5y9wGJ3bkbUeuO31PhZEXCnFQJAIqmbM0X8pVg0qZ+BLGRBOT7wlY0wAFWkTdkKVrQu5Ztv2zVIRfLu35Br4Et3tR/M8gS7wYe46mxdutI6MFyRIlJtn6VE2SSMkIoVRmrQEoTp2si83QbfZCqQ0CI97ZSvDmStflZTagJcixGzzTKQhfKzpmntJIzUICUgXHHJWfetWG9IEzK74QyqqDAaJkmoibfDJau4nOAzSxRSEYNsFMU6msNF6SRpBAH0v5kXsiSZ6xQlCVvSqNTozS5fYGIQuBntzRV29NIfFbtONHkDDkYn3bIuCmWwuLn3PfuC68PeJ4TqC6f+vFPCxXzPDOmxf83SEK5/9X/ayx/8R+M/KZpCSSn8m1g6USDXeqQ0COVSdYJWDIQVzKjJ75VClPNY5vkbauQhEr2d0K3rZ+jmbgnTMqISgsA6NFL7uJYR9mB77TEdejPL7KKlYu6D1oZ+CCRSruzKFYhG/BRTMS9qMTw21aj1LEtysrsMgi5XYNXmMEaoOV+Hosx+wLxEFhkTkgpWKQtz8ZRkEcPVKkz5cZUQf6sM6aWy+edwwda0Cl98q/NRHxXJa6eImo6B+vzOtDwq3OEhZldWzUwXPbV6AEFy2qOnIfAtAXFo2GlK6YJuMrgQCs9NNgQSaBmpsCJztUI4xGUdGyYu5Y8aF10tB9yUy/Jk5JU99bAel0g7WczQgnKyUYGXQ7q58q54iDiAZfvYFHYeQR6BU4mx/04IKCP1FgCXP1/AG3w7gJ6mV8hWhxF7a2p8mo1+sPsj6CFvcErhC9q0Lsd5y3ppSrKi2tK8BbAaOLfSJ4ETFFyn7EjBc9MVJ8zG2NAjenCRTbbaKb50Zr+QkytJiv0PpVLA2VsS5rKCc/sPZZkhjvTqUPDl0a+nGCfACPEKw8+mSkf8FenpU/s6tEg3fkp0DKF1fVkKsye7VNs9m7E5x/Te5570ICDsnpqfz5JiO3U6akjcv8k5Sfm0mxkZLrt4UZKGQDtKRHXxV8haolgvO5WGauCtZxZHMCtcqqNmMxyBGZZ7DtUf9A4zgkV73kYVmLCMWuFJsoBHegNDKuhvNMAZE/ayvhAn7Qa9Z25y7np4TRVTI27o3Q+qnLKeGLjP9tVTUTTHwtfKCF3N8m+kA2l044QkGv/XFtt0NT1mp/h2gePn6ZKr/DMOMex7WxWtL8oK2HizHriMS7IhrbXBAUkNVB6tteTHXm6TPUbWzZbJFw3weuyIKDfsyxfhJvwfvQIG6m9h/2cupA0xBy7QEp5TJYFNswvCU0DofcSI92FDmuOWUbn3ah/yXdlrMxPlY8KPPYAHpE+bNEkqYV+Hs3/Blhp8IE9GaHMSU5DKARGYolSxNatGh5CC69lQlj8eqYLzZrcpo4DZP97TSPTOx+vySTkwqMbr/YPPi2GdIJae3jZRYSZEYDK031Rd8/Skrref1QrgfUM6nPRRkNm8UkaUDtsybhv9hR823UXyvPhZdEsk3mi8fll764EgTtyewQ7V+SCEgXEyYblOYxD4oJ7aikF1mYE3mJRIkhV8y4BsbUpvCKw15hE+U8s36eErgFuhFtuj9UxwMNr6JoQCA236qwNN5nYrHsXLkFhnCwqoMZ2XL8wQWpYAXTR4IiLjO4sRq1jfgo91jcFPpsOmmT1CBHS7QRoZm8SIvRqoo8IM07+0B6ehh3xsq9NnBLNUpKBtK5vTrKK+FuSIKqUS0kvNj5JSbDnDL1Mhi6DFIJA7EvLcKKcbPZ/jGAoToek4P94oQiHtbKy6vzXqlJxuRFy36PGWUmlFcl8ecU9Irulevqj6lJ0crGRLJEVN0JAf9AFq1k2NBI0B3OTcxdWgGfg9VbSCRx10lkvsE5l931oPAtJ2UU6NgZYyaHhSh5vqDjnu62qD5p7YSjnWh+zHcbzJbNP7PfwsqF9BztBzT57P/n4luDxr9rnbtXoaqPRznRIsMA4P0JzFxZw6/VMU8FIfp1F6dzXnhnYs9BLWL2SCspNBQ2b2qkxL+HKUCMANH3vQuvYEwxXbp5BgXmm4dcxtvkKa0ynXThlA/R/jxkrKRKDQfvKaZNCTRYvaUF3PSoIkcBavjtDKeQX6R3OwYUnv+xrwtZ7pZ1cQn+5/+ib2wv956a/uWZoYq2G0aP1I3RM7h2ldA5k2bXam5pcvi7VGgRDqoV5DhCqygncLGt7+zkbHOHK79sbzO5DsfrcK0RxqYVrmmfVOz2c/oYGZQyjSs7M8hs6OtwLhFCx4GUeFlPi22V4feuwd5GwcSXPJWv9YREx04Lr/Ni9JWHshJxCJDjcsBejXwkbigE8AzQzrxS47TGPBIro7FgkoFg3Uj4jeXedohGRFqgO147CCRUBFyi+57imJ6txxg6jwRx4/EQTxOtyXfcWSCerOiMozrY1IJMMtAEuf8p1XiJu97HfWBjV+wRVfHX5dqnl+RqBEDiSgAYZpLEPYmGgRljXgJnYuB+hg8U+cwBAi03gYfsiQlN5k4noRYCGEXvcHMZ7YeRK017fO1Y8AKea2gA9aaXifIz3RqZdebaXcW3wC1gvU49ukrbhalK7rXcq4kN1NRUkuAJ/X1aJZM4DoxJt7DxRBn1+CYAGuvuAnUQ2f+hG82uYStpYSjBQAIpgTa1wXCZ7FKe+4UfXlXjFn1YMM2/dm2KCVqqTHtFSsqsIfes9Z7N2gj6JOETCaNtUTABFYVDitXG1T9+Nu9r5UXeqj+E/fqD3V5AAEPHOlfV9CvxB/d1y6Rspcp5TxjquIpYfva/k92kHLprm5wbzinDtWRhpz7wMHtC3VUJUOuH0oP7F56Ckpy3fQYT4zWe7iJSZx+W9t98pzcK8YVuY2p+akmeXIMJaqG3E/GkCotc/UX9eTXZ9xJD24yVizYwtO0RvH8kjh8ahqZzPRUIFx5YINObPeB+cWo3m9f0ye6lorQ3B2xA2hC+j/9oDQyfBQsJzC60dsE64bTVu8kRNevNrgEi3kl6Dxm3qjydVpDxeEA0UbOLkngWmRkYMGR5E8GsQsZgS0YFGbAnJflVC8fhNXoSDmy1zUCDm4++iOahwIZWqwUdhCqjzhiIPiG6sxxuwEysEAyiRv8LYB49Z7t3viNviHX8Cva9tRX138LET409jU37SCHg5yEi6IylXlWwpX7jT9ef0UfG78PmTyP6oTHE7pjC3/7lRut99rZA0y7CaGZdRyNcKEym6EAT2Xff9Jphkrm1z2jKx7WPJZl0bCf76nH152vNWkxlpwDPgJdJaMjqJCoHY3ILIGY5RkSP9KKOdYatPtoPBRuhgnBuLWNmih9dSkxdSZBPvVQ3oARuKthyqYzS6Nh5f002FPDIIv4TQo0INPmx9pGxIh8Vajqbtlxfok2qi0Ma4N++ulZua5M/fxOys0nkAd6VBQJXEUIVl38Qne4+44CK9LB5KWd7EPm+Psvrquzotyl5RYqgpEA2hjAvIzsf7jouVCjvy9bAfWGHJIoWwLfAvZYE8dL5Id1gERST3xBrIgQC59shtclxQBI2DijURSh5Q0cwTO1nXHmOgctVcxcB5/wsQHBJBr0DNEUlHdjqrLAO0LX8aZxwZ9Ft2uvp9AmTKuOAascH3MWWL1xgDPubimnn6QXf1suZQEyT4GlR48tg9idf23qUnzUWAR6PzN+y0ZBkLHpoQh0e4f7RtGS6hSpd/ZUcQ4O2jLB5A8wk5QIllyz0to2fOmtCpdMPbVT0B9IK0EdqFcxV6FYCPAoMRJvbK98EzP4lEQHEe/XzONydOzzUe+f4wzgWfmYmrcGCzIi5CfQ3JptaVtLu5s4Q2//dbF0tO3qTPx6DZQ3vPnwCQBA6r6e4pc8z/vSl9L3Kj264OGQsY7R+N1o8RjKnuUlcuu6bi6s+Ac1JpOg4ZciJ8Xoy7kXJ6Dy9iBFzNNLEgoAamJYNxGEq142kACpRaYxNFBsfUBeuiy0aN9+ufgfUtFNTkq3hICYAjSLwmt8Al7e20C93zide3SMvJh1pACgz7yJrxoe3PzBQxWSc4Rui5Vc4X+wcZn/VzbeK1Ou9ATJmId/XYCCmmsns+1IOS97UQ49xCf6Twy8F16yT50eqEkZtIKqKOdTeWwCbmfYUHPDY+XG6KwRfKXzOCA+DiI4KzsUx+i0+4FIf9xiPXAc/Wh4T7YFYggPHPeUCMD2U7AbyTvNd9e0uJkQ/FFakjpNAb+xZBrlNmqkTN+QqbngZADVtWOt3Y3focxmWisAAAA=",
    "side_house.jpg": "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",
    "project.jpg": "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",
    "interior.jpg": "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",
    "f_cost.jpg": "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",
    "f_approval.jpg": "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",
    "f_consult.jpg": "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",
    "f_interior.jpg": "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",
    "t_modern.jpg": "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",
    "t_earthy.jpg": "UklGRiYJAABXRUJQVlA4IBoJAAAQNwCdASrwAGkAPpFAmUklo6gqqhPtGVASCWUG+IUEJVTP3esowXc8fJ7bHGtzBPIdrLiqRvJp1M3wRCOmqrF1Ivcf9XS7B9e+B16k+GaTpLdDpHP/BTvL3oxdqWyLga6N3s7HWsIuowyvBeTplD8+jou1LkyS7AvGUGaphbdmmcz63+lfRrOjVbbWKAuxOke68LxSucUjZLW3ijg/BAXu73fVkE3bsSQEuzyRDVkcXlJ8K4BwA3X/RdKKMJ8NqG/wtU24T68NqRXRmzGXDq2VpxXVTBb0zM0I40+X2K/By8NN+g1OXCINSY+GLxjs9BTFbx93VSKEXkHLJHmySQLwnz2YVBPzq1Pg1ulvgHZ9w1slrEIDhyGB7r4smt2La/U8damUlMdUtkm2OQda6uEmkB/L9udU9hWXk9oPwfjlPMja8nj/6KVl14l/kGW5Jv/jjMHNxEKnUbI9MgtKilOM+l5O8e0NddicEVER+06OLa1qfGlti+AMRJo3zNQbRwH5P0OiGeZEWW+1z083DGX2m3SVJLYEd6SSDvYLDdLEeoI9a6hiP/hTb+qfaLCpAeHqnZu1ikEzA/XLH6rh3Xj6AAD+/nwdxZyreqGGog+4jD4rEQDjQ2ikNJmHAZIQ/xl+ubl1XxnMzHgxgMsVi+MIHINWzO61yB5ODjAs88Op+Ls1lzjBNpBVM4mkRktzxH/GycneDOtz/hm7ruIDQ5CtWqxknke136/GGVmBrbIi6e/KLagDod1dGFHWQLC1MXuNRM0IQekE0qvYfFjLnQkPSZZEC9Xp2HWGO68aMYSL8QUCsmlaeC6RQWIHvgG2KAIdKRLPd81Hy7f5SBf/jbYWHV27wE0Y3o7+zY4oScHWyz7NCbHqk7JZoFeX+gAayL84Na0V6QZJnr5WCBRSHji83UTYpJzOKnjcOrj4ww3h1HVKQ6TQU6TUT4OaeCq/lQrENSOHNvkGFAaOIC8r1usBvn2NrU/meJh2e4UxKBvVkitX2Bx4Id5PSM1sIZdKQx25RwAOt/OHvx0ROXGQCc0mYmDlll70TrLCtLYmN/3q2Gvlj0Gpy/BuLHLGd6V/aK528WcjYJlHQqflyRZ42RArdlWbL7uof14bENNGqRtxV56HGRIkhQu4RUUHKCHTXcX06RFoUdUiwZAeTxcFQNSOv2zSjV64x+PxIgm5xtX1gXYfMJkHQ/RGeyX02zajg/A/pdfm/V2KvRexKvOxrCkuDJu67FknSVonoTsOpbqL0GDpA7Sl3ykT40W02YDvsqq5vfJNop5ZPjMKpCJz4Odb8uK7DL9ZevZopG8gLKmC0kZGmr4Kl4PH5cC1i9Adh642C+w/Yl4Nf77KNPg6MISmDDyQ5r2UwLJIM1znWlciF5wE/zXbqPjeRiS+A3jhxqSZ5dCcw39xg5uvALjgaYcEW1Sz8xuEWoNQV+n366gcPRrztwTvYhIMRsCT84i//MuM96XwMlES2oxRhtKNOYFHGv7GMiB7FNea++n0Kw5aHjVyDd/CT7fTkE1nSd8gGdl0Hbrz6nS264oXwZjp7mJjqYt4sna/e74/8nFGNvLkzCmmpzV0DkYnGf/m6Mukn8BC8GwkijVqs9FaVMtaAL0ycQjsYR6jFa/3RVawi7bOVHxUn48MtQN6/Dqu+wHvPZ9Ux+afvMHZ6EFF3JybqoMD+0X4/XwW8/lpA4adhqRf3P+ifEjg+dE/PX1DzgZ6H3lv7CEkrFrgKiqA9fvysmiS2a6/LfelRh0lzLcxpBL6gaiybhCEaywDycQ44YwBKCPlDAGWyGi+u3JK1UfzqW43iN8w9fgpczvVLmOIrexgdqAyJ1gNRH4U0QQ+pFNOTMZtaoqVXye227W3bEO/UWWWY8BiqQOltRlAHNtf5HozzxVaK8pRq93zMzO+MPYS08Oqus/t0bjUaGPKedqDcPSss5XbX4+7vxzCCZlOx6zecyc8RwT2w/gogLEGV9hT9RSLtPr5HUGghZ2nh5ec4wrF8J1FRnEQ3TguS9ns8Dn1yFDJd8QpCSBAQBsizWPr3XZ9K7b7+x3/HOI6MDbrd/dsROzN5XftD4h+K9RgYr2ro0KPIsRGz4ApSeMSVXZaV4/Mi1Tp2QybmLE1Heihmidvj0NSHrgcEGg8kw0njVuKyG8QXJcr2hR/ACl3LEz+v4T1NyOupN893mmdow9dcGvWIVlG5+upmkY+hiIRwdNusSLxWZHacVzQCafZGJ5Kqp9Gk6Z91WrxS0FH3fIkJdKXYLs70r0Pj5eGGApXUViHcKoocmEafmVULJyu3bmPolc8zoF8VVBsHTY3KAwOp0t0otCFgiqBU7xjGD0mjsukSDObxE4g1J0akZYcWqxei2icf4YeTApdpH/vseNsGqiownOqMuqeGxcb2EFo3bNaaAzG+zI5rU8Wm4buC1nGVDWiZGuF+4S/f3d7Ohzwp7Zv2xAiL4VcaS103t3oYo+oqj7j2n/JsNJVMLy5qvAhbcVQemPO6GL8m3nA8bdXrT5n5kUW7/bMdlyZ3jMWZ5ph5t0N8bQF0HgZQEt5KZbp5fOmEmapXJ2Rvu9uURkX94J7lHiGbMMblOeOzYZPlRQqpfJE2u1WIEHojNm69h76myq0i1IRK3x7i2/0AWE5aitcanq8YVD2e2vGBB1p3aTIabNpNrWtXqUrEGvLbI5nrbrk0N8XufDl4atFgwdMl9U23YGuc6jDvX6xlZyemLMb3nT4CXR2uLcr6+Ug/atDpA8XP5ud3N7gb0fDkC9iv4K+FhA+lER13mQDHQRJ3aMmRLQFgrXd+g6QxJuyOHz/hpIw2RBBk5p0sI12g116PcnJGjKLpUhF7dA9wIqy1EX6mX0iXh0B3dXLyZWre36RwkEvt5SbzmksP9KT7FzBh/gFONgq2It784+kiL0MZRY6cyejP56ojQO2EdjYovG+9S+m9otz2LzYcQ9qrNS0uuic26h8+5leOBu2//VT49s5iAMcnvYm8Yc5sFT5T9cWi4B8X3zzqLd7tqaehV1MFKmpd4EU2+3viO5EToLHfT9pTNNgLAy0mox0zAs7roJjKUgYaJHfva9JDK3caGo3jObwOHuYfOBnwcAAAA==",
    "t_classic.jpg": "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",
    "t_japandi.jpg": "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",
}


if __name__ == "__main__":
    if st.runtime.exists():          # started with `streamlit run app.py`
        st.set_page_config(page_title="BuildWise AI", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")
        main()
    else:                            # started with `python app.py` -> launch Streamlit for you
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__, "--theme.base", "light"]
        sys.exit(stcli.main())
