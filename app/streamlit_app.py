"""Interface web local (Streamlit) do reconhecedor facial - Parte 2 (spec 005, ADR-004).

Uso (na pasta do projeto, com o ambiente ativo):
    streamlit run app/streamlit_app.py

Tudo roda localmente: nenhuma imagem sai do computador.
"""
import io
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # caminhos da configuracao (data/private, logs) sao relativos a raiz do projeto

import streamlit as st  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from facerec.appfactory import build_service, setup_logging  # noqa: E402
from facerec.config import load_config  # noqa: E402
from facerec.detection import load_image  # noqa: E402
from facerec.errors import FacerecError  # noqa: E402

CONFIG_PATH = Path(os.environ.get("FACEREC_CONFIG", ROOT / "configs" / "app.yaml"))
IMAGE_TYPES = ["jpg", "jpeg", "png"]
log = logging.getLogger("facerec")

st.set_page_config(page_title="Reconhecedor facial", page_icon="🙂")


@st.cache_resource(show_spinner="Carregando os modelos (na primeira vez demora um pouco)...")
def get_service():
    cfg = load_config(CONFIG_PATH)
    setup_logging(cfg)
    return cfg, build_service(cfg)


def annotate(image, matches):
    """Desenha a caixa e o rotulo (nome ou 'desconhecida') sobre cada face."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=max(14, image.width // 35))
    width = max(2, image.width // 200)
    for m in matches:
        color = "#1f9d55" if m.recognized else "#d64545"
        if m.recognized:
            text = f"{m.name} ({m.score:.2f})"
        elif m.score is None:
            text = "desconhecida"
        else:
            text = f"desconhecida ({m.score:.2f})"
        draw.rectangle(m.box, outline=color, width=width)
        draw.text((m.box[0] + 3, max(0, m.box[1] - font.size - 4)), text, fill=color, font=font)
    return out


def page_recognize(service):
    st.header("Reconhecer")
    if not service.has_people():
        st.info("Nenhuma pessoa cadastrada. Toda face será classificada como desconhecida "
                "até você cadastrar alguém.")
    source = st.radio("Origem da imagem", ["Câmera", "Arquivo"], horizontal=True)
    if source == "Câmera":
        file = st.camera_input("Tire uma foto")
        if file is None:
            st.caption("Se a câmera não aparecer, permita o acesso no navegador ou use a opção Arquivo.")
    else:
        file = st.file_uploader("Envie uma imagem", type=IMAGE_TYPES)
    if file is None:
        return
    try:
        image = load_image(file)
        matches = service.recognize(image)
    except FacerecError as e:
        st.error(f"Não foi possível processar a imagem: {e}")
        return
    except Exception:
        log.exception("erro inesperado no reconhecimento")
        st.error("Erro inesperado ao processar a imagem. Detalhes em logs/app.log.")
        return
    if not matches:
        st.warning("Nenhuma face detectada na imagem.")
        return
    st.image(annotate(image, matches), use_container_width=True)
    for i, m in enumerate(matches, 1):
        if m.recognized:
            st.success(f"Face {i}: {m.name} (similaridade {m.score:.2f})")
        elif m.score is None:
            st.warning(f"Face {i}: pessoa desconhecida.")
        else:
            st.warning(f"Face {i}: pessoa desconhecida (melhor similaridade {m.score:.2f}, "
                       "abaixo do threshold).")


def page_enroll(cfg, service):
    st.header("Cadastrar pessoa")
    enr = cfg["enrollment"]
    st.write(f"Use de {enr['min_photos']} a {enr['max_photos']} fotos da mesma pessoa, com uma "
             "única face em cada, de frente e com boa iluminação.")
    fid = st.session_state.setdefault("form_id", 0)
    camera_photos = st.session_state.setdefault("camera_photos", [])

    name = st.text_input("Nome", key=f"name_{fid}")
    consent = st.checkbox(
        "Confirmo que a pessoa é maior de 18 anos e assinou o Termo de Consentimento "
        "(Anexo A), guardado fora do repositório.", key=f"consent_{fid}")
    uploads = st.file_uploader("Fotos (arquivos)", type=IMAGE_TYPES,
                               accept_multiple_files=True, key=f"upl_{fid}")
    snap = st.camera_input("Ou tire fotos com a câmera (uma por vez)", key=f"cam_{fid}")
    if snap is not None and st.button("Adicionar esta foto da câmera"):
        camera_photos.append(snap.getvalue())
        st.rerun()
    if camera_photos:
        st.caption(f"{len(camera_photos)} foto(s) da câmera adicionada(s).")
        if st.button("Limpar fotos da câmera"):
            camera_photos.clear()
            st.rerun()

    if st.button("Cadastrar", type="primary"):
        if not consent:
            st.error("Marque a confirmação de consentimento antes de cadastrar.")
            return
        try:
            images = [load_image(f) for f in uploads] + [load_image(io.BytesIO(b)) for b in camera_photos]
            service.enroll(name, images)
        except FacerecError as e:
            st.error(f"Cadastro não realizado: {e}")
            return
        except Exception:
            log.exception("erro inesperado no cadastro")
            st.error("Erro inesperado no cadastro. Detalhes em logs/app.log.")
            return
        st.session_state["form_id"] = fid + 1
        st.session_state["camera_photos"] = []
        st.session_state["flash"] = "Pessoa cadastrada com sucesso."
        st.rerun()


def page_people(service):
    st.header("Pessoas cadastradas")
    people = service.list_people()
    if not people:
        st.info("Nenhuma pessoa cadastrada.")
        return
    for p in people:
        left, right = st.columns([4, 1])
        left.write(f"**{p['name']}** ({p['n_embeddings']} foto(s) de cadastro)")
        if right.button("Excluir", key=f"del_{p['id']}"):
            st.session_state["to_delete"] = p["id"]
    to_delete = st.session_state.get("to_delete")
    if to_delete:
        st.warning("A exclusão remove o cadastro, os embeddings e as fotos armazenadas dessa pessoa. "
                   "Não pode ser desfeita.")
        if st.button("Confirmar exclusão", type="primary"):
            try:
                service.delete_person(to_delete)
                st.session_state["flash"] = "Pessoa excluída."
            except FacerecError as e:
                st.session_state["flash"] = f"Exclusão não realizada: {e}"
            st.session_state["to_delete"] = None
            st.rerun()
        if st.button("Cancelar"):
            st.session_state["to_delete"] = None
            st.rerun()


def page_about(cfg):
    st.header("Sobre")
    rec = cfg["recognition"]
    st.write("Reconhecimento facial local: detecção (MTCNN) → embedding (FaceNet, VGGFace2) → "
             "comparação por protótipo com similaridade de cosseno → threshold.")
    st.write(f"Threshold atual: **{rec['threshold']}** (configs/app.yaml). Similaridade abaixo disso "
             "= pessoa desconhecida.")
    st.write("Os dados ficam só neste computador (data/private/). Não são enviados a nenhum serviço.")


def main():
    try:
        cfg, service = get_service()
    except Exception:
        logging.getLogger("facerec").exception("falha ao iniciar a aplicacao")
        st.error("Não foi possível iniciar a aplicação (modelos ou configuração). "
                 "Confira o ambiente e o arquivo configs/app.yaml.")
        st.stop()
    page = st.sidebar.radio("Menu", ["Reconhecer", "Cadastrar", "Pessoas e exclusão", "Sobre"])
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash) if "não" not in flash.lower() else st.error(flash)
    if page == "Reconhecer":
        page_recognize(service)
    elif page == "Cadastrar":
        page_enroll(cfg, service)
    elif page == "Pessoas e exclusão":
        page_people(service)
    else:
        page_about(cfg)


main()
