"""StudyCompanion v2 - an ethical, explainable study advisor (Streamlit).

Run with:  streamlit run app.py   (needs streamlit >= 1.37)

HCAI principles implemented:
  1. Onboarding & limitations  - welcome dialog, reopenable, plus a persistent limits note
  2. Privacy                   - local-mode banner, "Reset my answers", nothing is persisted
  3. Bias mitigation           - behavioral inputs only, explicit list of what we do NOT ask
  4. Guidance / transparency   - every input has a plain-language guide
  5. Confidence                - ranges, confidence meter and the reasons behind it
  6. Explainability            - SHAP-style bars, "fixed vs. changeable" labels, counterfactual advice
  7. User control              - what-if explorer, human-in-the-loop reminder
  8. Feedback                  - thumbs plus optional comment
All numbers come from a mock linear model, not from trained weights.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

st.set_page_config(page_title="StudyCompanion", page_icon="🌱", layout="wide")

st.markdown(
    """
<style>
.stApp { background:#f3f7fa; }
.shield { background:#e1f5ec; border:2px solid #3aa675; color:#1f5f44; padding:.6rem 1rem;
          border-radius:12px; text-align:center; font-weight:600; margin-bottom:1rem; }
.fairness { background:#e3f0f9; border-radius:12px; padding:.7rem .9rem; font-size:.88rem; margin-bottom:.6rem; }
.limits { background:#fbe9e7; border-left:4px solid #d9695f; border-radius:8px; padding:.6rem .9rem; font-size:.88rem; }
.rangebox { text-align:center; background:linear-gradient(135deg,#e3f0f9,#e1f5ec); border-radius:16px; padding:1.4rem 1rem; }
.rangebox .lbl { color:#5f7080; font-size:.85rem; text-transform:uppercase; letter-spacing:.05em; }
.rangebox .rng { font-size:2.3rem; font-weight:700; color:#243441; }
.rangebox .rng span { font-size:1.1rem; color:#5f7080; font-weight:500; }
.meter { height:10px; max-width:260px; margin:.4rem auto 0; background:#fff; border-radius:99px; overflow:hidden; }
.meter i { display:block; height:100%; background:#3aa675; border-radius:99px; }
.note { color:#5f7080; font-size:.85rem; text-align:center; margin-top:.8rem; }
.row { display:grid; grid-template-columns:minmax(120px,1.2fr) 2fr minmax(80px,.7fr); gap:.6rem;
       align-items:center; margin:.5rem 0; font-size:.92rem; }
.track { position:relative; height:18px; background:#e4ecf2; border-radius:6px; }
.track::after { content:""; position:absolute; left:50%; top:-3px; bottom:-3px; width:2px; background:#5f7080; opacity:.5; }
.bar { position:absolute; top:0; bottom:0; }
.bar.pos { left:50%; background:#3aa675; border-radius:0 6px 6px 0; }
.bar.neg { right:50%; background:#d9695f; border-radius:6px 0 0 6px; }
.val { font-weight:700; } .val.pos { color:#3aa675; } .val.neg { color:#d9695f; }
.tag { font-size:.7rem; background:#e4ecf2; color:#5f7080; border-radius:99px; padding:0 .45rem; margin-left:.3rem; }
.axis { display:flex; justify-content:space-between; font-size:.78rem; color:#5f7080; }
.tip { background:#e1f5ec; border-left:4px solid #3aa675; border-radius:8px; padding:.65rem .9rem; margin:.5rem 0; }
@media (max-width:600px){ .row { grid-template-columns:1fr; gap:.2rem; } }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- model
BASE = 10.0


@dataclass
class Feature:
    key: str
    title: str
    group: str
    kind: str  # "slider" | "toggle"
    guide: str
    effect: Callable[[float], float]
    default: float
    lo: float = 0
    hi: float = 1
    step: float = 1
    unit: str = ""
    changeable: bool = True
    improve: float = 0  # signed step to try in advice
    extreme: Optional[Callable[[float], bool]] = None
    advice: str = ""  # {gain} is filled in

    def show(self, v):
        if self.kind == "toggle":
            return "Yes" if v else "No"
        return f"{int(v)}{self.unit}"


FEATURES = [
    Feature("last", "Most recent test grade", "Study habits", "slider",
            "Your latest grade on a 0-20 scale. Earlier results are the strongest single signal in historical "
            "data. Use your real grade, or leave it at 10 if you don't know.",
            lambda v: (v - 10) * 0.55, 10, 0, 20, 1, "/20", changeable=False,
            extreme=lambda v: v <= 3 or v >= 18),
    Feature("study", "Weekly study time", "Study habits", "slider",
            "Hours per week you study outside class, including homework. Count focused time only. "
            "Returns diminish after about 15 hours.",
            lambda v: (min(v, 15) - 6) * 0.22, 6, 0, 20, 1, " hrs", improve=+2,
            extreme=lambda v: v <= 1 or v >= 18,
            advice="Adding 2 weekly study hours could increase your score by about {gain} pts."),
    Feature("abs", "Absences", "Study habits", "slider",
            "How many classes you missed this period (excused or not). Missing class is one of the most "
            "changeable factors.",
            lambda v: -(v - 4) * 0.09, 4, 0, 30, 1, improve=-2,
            extreme=lambda v: v >= 20,
            advice="Reducing absences by 2 days could increase your score by about {gain} pts."),
    Feature("fail", "Past failures", "Study habits", "slider",
            "Number of courses you didn't pass before (0-3). This is history, so we show its effect "
            "but never advise you to 'fix' it.",
            lambda v: -v * 1.0, 0, 0, 3, 1, changeable=False, extreme=lambda v: v >= 3),
    Feature("higher", "Aims for higher education", "Motivation", "toggle",
            "Do you plan to continue to further study after this program? Motivation shows up in the data, "
            "but it is a goal, not a requirement.",
            lambda v: 0.8 if v else -0.4, 1, changeable=False),
    Feature("activities", "Extracurricular activities", "Motivation", "toggle",
            "Do you take part in a club, sport or other activity? Structured routines tend to help.",
            lambda v: 0.2 if v else -0.1, 1, changeable=False),
    Feature("free", "Free time after school", "Lifestyle", "slider",
            "1 = almost none, 5 = a lot. A moderate amount of rest supports learning; both extremes matter.",
            lambda v: -abs(v - 3) * 0.12, 3, 1, 5, 1,
            extreme=lambda v: v in (1, 5)),
    Feature("out", "Going out with friends", "Lifestyle", "slider",
            "1 = rarely, 5 = very often. Social time is healthy. The model only sees a small dip at the high end.",
            lambda v: -(v - 3) * 0.25, 3, 1, 5, 1, improve=-1, extreme=lambda v: v == 5,
            advice="Going out one step less often could add about {gain} pts. Social time is still important, "
                   "so treat this as a trade-off."),
    Feature("health", "Current health", "Lifestyle", "slider",
            "1 = very poor, 5 = very good. If your health is low, please prioritise it: no grade is worth more.",
            lambda v: (v - 3) * 0.1, 3, 1, 5, 1, improve=+1,
            advice="Improving your wellbeing by one step is linked to about {gain} pts. Sleep and movement are "
                   "good places to start."),
    Feature("dalc", "Alcohol on school days", "Lifestyle", "slider",
            "1 = none, 5 = very high. Answers are never stored. Be honest, since the model only helps when "
            "inputs are real.",
            lambda v: -(v - 1) * 0.3, 1, 1, 5, 1, improve=-1, extreme=lambda v: v >= 4,
            advice="Cutting back on school-day alcohol by one step could add about {gain} pts."),
    Feature("walc", "Alcohol on weekends", "Lifestyle", "slider",
            "1 = none, 5 = very high. Weekend habits can affect Monday focus.",
            lambda v: -(v - 1) * 0.12, 1, 1, 5, 1, improve=-1, extreme=lambda v: v >= 4,
            advice="Cutting back on weekend alcohol by one step could add about {gain} pts."),
]
FMAP = {f.key: f for f in FEATURES}

NOT_ASKED = ["Gender", "Age", "Nationality or ethnicity", "Parents' education or job",
             "Family income", "Home address or internet access", "Relationship status"]


def score(vals):
    return min(20.0, max(0.0, BASE + sum(FMAP[k].effect(v) for k, v in vals.items())))


def confidence(vals, total):
    reasons, c = [], 84.0
    ext = [FMAP[k].title for k, v in vals.items() if FMAP[k].extreme and FMAP[k].extreme(v)]
    if ext:
        c -= 4 * len(ext)
        reasons.append("Rare values in: " + ", ".join(ext).lower() + " (less historical data).")
    if abs(total - 10) > 4:
        c -= abs(total - 10) * 0.6
        reasons.append("Very high or low scores are harder to predict.")
    if vals["last"] == 10:
        c -= 4
        reasons.append("No last grade was provided, so the estimate is less precise.")
    if not reasons:
        reasons.append("Your inputs fall in a well-covered range of the historical data.")
    return min(90.0, max(55.0, c)), reasons


def predict(vals):
    total = score(vals)
    conf, reasons = confidence(vals, total)
    m = 0.8 + (100 - conf) / 20
    return total, max(0.0, total - m), min(20.0, total + m), conf, reasons


def fmt(n):
    return f"{'+' if n >= 0 else '−'}{abs(n):.1f}"


# --------------------------------------------------------------------------- onboarding
INTRO = (
    "**What it is:** A study advisory tool to help you optimize your habits.\n\n"
    "**Limitations:** This is a statistical model based on historical data. "
    "It does not define your worth or guarantee future results.\n\n"
    "**How to use it:** answer the questions on the left (each has a short guide), press "
    "*Calculate Prediction*, then explore what drives the estimate and try what-if changes.\n\n"
    "**Your data:** everything stays in this browser session."
)


@st.dialog("Welcome to StudyCompanion 👋")
def welcome():
    st.markdown(INTRO)
    if st.button("I understand – let's start", type="primary"):
        st.session_state.onboarded = True
        st.rerun()


if not st.session_state.get("onboarded"):
    welcome()

# --------------------------------------------------------------------------- header
st.markdown(
    '<div class="shield">🛡️ Local Mode Active: Your data never leaves your browser. '
    "No data is stored or transmitted.</div>",
    unsafe_allow_html=True,
)
h1, h2, h3 = st.columns([5, 1.2, 1.2])
h1.title("🌱 StudyCompanion")
h1.caption("Small habits, steady growth. Let's explore what could help you most.")
if h2.button("About & limits", use_container_width=True):
    welcome()
if h3.button("Reset my answers", use_container_width=True, help="Clears everything from this session."):
    st.session_state.clear()
    st.rerun()

left, right = st.columns([1, 1.5], gap="large")

# --------------------------------------------------------------------------- input form
with left:
    st.subheader("1 · Your study habits")
    st.markdown(
        '<div class="fairness">ⓘ To ensure algorithmic fairness, we only ask for actionable behavioral data, '
        "not personal background.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("What we deliberately do NOT ask"):
        st.markdown("\n".join(f"- {n}" for n in NOT_ASKED))
        st.caption("These can encode unfair bias, so the model never sees them.")

    with st.form("habits"):
        answers = {}
        for group in ["Study habits", "Motivation", "Lifestyle"]:
            st.markdown(f"**{group}**")
            for f in [x for x in FEATURES if x.group == group]:
                if f.kind == "slider":
                    answers[f.key] = st.slider(f.title, int(f.lo), int(f.hi), int(f.default), int(f.step),
                                               key=f"in_{f.key}")
                else:
                    answers[f.key] = int(st.toggle(f.title, bool(f.default), key=f"in_{f.key}"))
                st.caption(f"ℹ️ {f.guide}")
        submitted = st.form_submit_button("Calculate Prediction", type="primary", use_container_width=True)
    st.markdown(
        '<div class="limits"><b>Remember:</b> this is a statistical model based on historical data. It does not '
        "define your worth or guarantee future results.</div>",
        unsafe_allow_html=True,
    )

if submitted:
    st.session_state.vals = answers
    st.session_state.feedback = None

# --------------------------------------------------------------------------- results
with right:
    vals = st.session_state.get("vals")
    if not vals:
        st.subheader("2 · Your insights")
        st.info("🌱 Fill in the form on the left and press **Calculate Prediction**.")
        st.stop()

    total, lo, hi, conf, reasons = predict(vals)
    st.subheader("2 · Your estimated outcome")
    st.markdown(
        f'<div class="rangebox"><div class="lbl">Predicted Range</div>'
        f'<div class="rng">{lo:.1f} – {hi:.1f} <span>/ 20</span></div>'
        f'<div style="margin-top:.8rem;font-weight:600">Model Confidence: {round(conf)}%</div>'
        f'<div class="meter"><i style="width:{conf}%"></i></div></div>'
        '<div class="note">An estimate, not a verdict. Your effort and choices matter more than any model.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Why this confidence level?"):
        st.markdown("\n".join(f"- {r}" for r in reasons))
        st.caption("A wider range means the model is less certain. Treat the whole range as plausible.")

    # ---- explainability
    st.subheader("3 · Inside the Algorithm")
    st.caption("How much each factor moved your estimate compared with an average student (10 / 20). "
               "Green raises it, red lowers it.")
    contribs = sorted(((k, FMAP[k].effect(v)) for k, v in vals.items()), key=lambda c: -abs(c[1]))
    mx = max(abs(c[1]) for c in contribs) or 1
    rows = ""
    for k, v in contribs:
        f, cls = FMAP[k], "pos" if v >= 0 else "neg"
        tag = "" if f.changeable else '<span class="tag">fixed</span>'
        rows += (f'<div class="row"><span>{f.title} ({f.show(vals[k])}){tag}</span>'
                 f'<div class="track"><div class="bar {cls}" style="width:{abs(v) / mx * 50:.1f}%"></div></div>'
                 f'<span class="val {cls}">{fmt(v)} pts</span></div>')
    st.markdown(rows + '<div class="axis"><span>◀ Lowers score</span><span>Raises score ▶</span></div>',
                unsafe_allow_html=True)
    st.caption("“Fixed” factors describe your history or goals. They are shown for transparency, "
               "but we never ask you to change them.")

    # ---- counterfactual advice: changeable habits only, ranked by potential gain
    def gain_for(f):
        nv = vals[f.key] + f.improve
        if not (f.lo <= nv <= f.hi):
            return 0.0
        return score({**vals, f.key: nv}) - total

    options = sorted(((gain_for(f), f) for f in FEATURES if f.changeable and f.improve),
                     key=lambda x: -x[0])
    tips = [f.advice.format(gain=f"{g:.1f}") for g, f in options if g >= 0.1][:2]
    fallback = ["Great work! Keeping your current habits steady is the best way to maintain your strengths.",
                "Try one short, focused study session per week and notice how it feels. Small steps compound."]
    while len(tips) < 2:
        tips.append(fallback[len(tips)])
    st.markdown("**What you could try**")
    st.markdown("".join(f'<div class="tip">💡 {t}</div>' for t in tips), unsafe_allow_html=True)
    st.caption("Effects come from a simplified model and may not apply to you. If a result worries you, "
               "talk to a teacher, mentor or study coach: a person knows your situation better than a model.")

    # ---- user control: what-if
    st.subheader("4 · Try a what-if")
    adj = [f for f in FEATURES if f.changeable and f.kind == "slider"]
    pick = st.selectbox("Change one habit", adj, format_func=lambda f: f.title, key="wi_pick")
    new = st.slider(f"{pick.title} – new value", int(pick.lo), int(pick.hi), int(vals[pick.key]),
                    key=f"wi_{pick.key}")
    t2, lo2, hi2, c2, _ = predict({**vals, pick.key: new})
    c_a, c_b = st.columns(2)
    c_a.metric("Predicted range", f"{lo2:.1f} – {hi2:.1f}", f"{t2 - total:+.1f} pts vs. now")
    c_b.metric("Model confidence", f"{round(c2)}%", f"{round(c2 - conf):+d}%")
    st.caption("This is an exploration, not a promise. Your answers above are not changed.")

    # ---- feedback
    st.divider()
    st.markdown("**Was this explanation clear and helpful?**")
    up, down, _ = st.columns([1, 1, 6])
    if up.button("👍", key="up", help="Thumbs up"):
        st.session_state.feedback = "up"
    if down.button("👎", key="down", help="Thumbs down"):
        st.session_state.feedback = "down"
    if st.session_state.get("feedback"):
        st.success("Thank you for your feedback!")
        st.text_area("Anything unclear? (optional)", key="fb_text", max_chars=300,
                     help="Kept only in this browser session. Nothing is sent anywhere.")
