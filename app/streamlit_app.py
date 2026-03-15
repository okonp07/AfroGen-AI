from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from afrogen import load_app_config
from afrogen.backends import create_backend
from afrogen.ui.state import ensure_delta_state


config = load_app_config()
app_config = config["app"]
latent_shape = tuple(app_config["latent_shape"])
backend = create_backend(
    name=app_config["backend"],
    image_size=app_config["image_size"],
    latent_shape=latent_shape,
    artifact_path=PROJECT_ROOT / app_config["trained_backend_artifact"],
)


st.set_page_config(page_title=app_config["title"], layout="wide")
st.title(app_config["title"])
st.caption("Prompt-to-portrait MVP with editable latent controls.")
st.caption(f"Active backend: `{backend.info.name}`")
st.caption(f"Backend load state: `{backend.info.load_state}`")

with st.sidebar:
    st.subheader("Generation")
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=7, step=1)
    prompt = st.text_area(
        "Prompt",
        value="A smiling young Black woman with braids, golden earrings, and a calm studio portrait",
        height=120,
    )

delta = ensure_delta_state(st.session_state, latent_shape)

left, right = st.columns([1.2, 1])

with right:
    st.subheader("Latent Matrix Editor")
    st.write("Adjust the latent matrix to push the portrait in different directions.")
    for row in range(latent_shape[0]):
        cols = st.columns(latent_shape[1])
        for col in range(latent_shape[1]):
            key = f"latent_{row}_{col}"
            value = cols[col].slider(
                f"{row},{col}",
                min_value=-1.0,
                max_value=1.0,
                value=float(delta[row, col]),
                step=0.05,
                key=key,
                label_visibility="collapsed",
            )
            delta[row, col] = value
    if st.button("Reset Matrix"):
        st.session_state.latent_delta = np.zeros(latent_shape, dtype=np.float32)
        st.rerun()

result = backend.generate(prompt=prompt, seed=int(seed), delta=delta)

with left:
    st.subheader("Generated Portrait")
    st.image(result.image, use_container_width=True)
    if result.backend_message:
        st.info(result.backend_message)
    st.write("Prompt profile:")
    st.json(
        {
            "age_group": result.profile.age_group,
            "skin_tone": result.profile.skin_tone,
            "hairstyle": result.profile.hairstyle,
            "expression": result.profile.expression,
            "accessory": result.profile.accessory,
        }
    )
    st.write("Base latent matrix:")
    st.dataframe(result.base_latent)
    st.write("Edited latent matrix:")
    st.dataframe(result.edited_latent)
