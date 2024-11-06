import streamlit as st
import os
import json
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import csv
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import base64
from pathlib import Path

csv_path = "csv_audit/component_defects.csv"

st.set_page_config(
    page_title="Home", page_icon="font/ icon.png", layout="wide"
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

def update_key():
    st.session_state.uploader_key += 1


st.markdown("""
            <style>

            .block-container
                {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    margin-top: 0rem;
                    padding-left: 10rem;
                    padding-right: 10rem;
                    
                }
            </style>
            """, unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

st.markdown("""<hr style="height:8px;border-radius:4px;color:#333;background-color:#3B6936; margin-bottom: 0rem;" /> """,unsafe_allow_html=True,)

coli,colh = st.columns([1,4], gap="medium")
with coli:
    st.markdown("""<img src="data:image/png;base64,{}" alt=" logo" style="float:right; width:140px; height:95px; margin-top:20px; margin-right:-60px;margin-bottom:-12px;"/>""".format(img_to_bytes("font/-logo-1.jpg")),unsafe_allow_html=True,)
with colh:
    st.markdown("""<h1 style='font-family:serif;color: #e9eef0; margin-left:60px;'> 
                    AI based Component Defect Inspection System
                </h1> """,unsafe_allow_html=True,)

st.markdown("""<hr style="height:8px;border-radius:4px;color:#333;background-color:#3B6936; margin-top: 0rem;" />""",unsafe_allow_html=True,)
# , -2px -2px 5px green
model_dir = "14_class_results_100Epochs2/weights/best.pt"
valid_image_extensions = {".jpg", ".jpeg", ".png"}  # 15_class_results_100epochs/best-15.pt
output_dir = "output_directory"
os.makedirs(output_dir, exist_ok=True)

classes = {
    0: "plating_peel_off",
    1: "contact_damage",
    2: "metal_chipp_off_moulded",
    3: "deflashing",
    4: "short_mould",
}

initial_defect_count = {
    "plating_peel_off": 0,
    "contact_damage": 0,
    "metal_chipp_off_moulded": 0,
    "deflashing": 0,
    "short_mould": 0
}

columns = ["Component"] + list(initial_defect_count.keys())
global defect_df
defect_df = pd.DataFrame(columns=columns)

def draw_boxes(image_path, predictions, output_dir):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font_size = 40  # Adjust the font size as needed
    cls_name = []
    for pred in predictions:
        # print(pred)
        for box in pred.boxes:
            # print(box)
            xyxy = box.xyxy
            # print(xyxy[0])
            x1, y1, x2, y2 = xyxy[0][0], xyxy[0][1], xyxy[0][2], xyxy[0][3]
            # conf = box.conf
            cls = box.cls
            # print(cls.item())
            class_name = classes.get(cls.item())
            if class_name not in cls_name:
                cls_name.append(class_name)
            # if type(class_name) is not None:
            #     # continue
            # print(class_name)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
            draw.text(
                (x1, y1 - 40),
                class_name,
                fill="red",
                font=ImageFont.truetype("./font/Arial.ttf", font_size),
            )

    annotated_image_path = os.path.join(output_dir, os.path.basename(image_path))
    image.save(annotated_image_path)
    return annotated_image_path, cls_name

# with open("csv_audit/defect_report.csv", mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(columns)

def write_to_csv(csv_list, path):
    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_list)


def main():
    # st.title("AI based Component Defect Detection")
    global col1, col2, col3, col4, col5, col6, col7, col8
    
    col4, col5 = st.columns([2,3], gap="medium")
    with col4:
        placeholder = "Enter Component No"
        text_input = st.number_input(
            "Component Number",
            min_value = 0,
            format = "%d",
            label_visibility="visible",
            disabled=st.session_state.disabled,
            placeholder={placeholder},
        )
        if text_input:
            component_name = (f"Component {text_input}") 
            # defect_df.Component = component_name
    with col5:
        
        uploaded_files = st.file_uploader(
            "Upload multiple images...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )
        # st.write(uploaded_files)
    col1, col2, col3 = st.columns([1,1,3], gap="medium")
    with col3:
        st.button(
        "Next Component",
        # mime="text/csv",
        on_click=update_key,
        )
        if st.button("Detect Defects"):
            # row_values = []
            model = YOLO(model_dir)
            # update_key()
            # csv_list = []
            audit_list = []
            for uploaded_file in uploaded_files:
                csv_list = []
                csv_list.append(uploaded_file.name)

                image_path = os.path.join(os.getcwd(), uploaded_file.name)
                # img = Image.open(image_path).resize((1080, 1080))
                # image_path = uploaded_file
                preds = model(image_path, conf=0.20)
                # annotated_image_path, c_name = draw_boxes(image_path, preds, output_dir)
                annotated_image_path, c_name = draw_boxes(image_path, preds, output_dir)
                csv_list.extend(c_name)
                audit_list.extend(c_name)
                write_to_csv(csv_list, csv_path)
                with col2:
                    st.image(
                        annotated_image_path,
                        caption="Image with Labelled Defect",
                        use_column_width=True,
                    )
            
            dl = set(audit_list)
            if  len(dl) == 0:
                html_list = f"""
                            <h3>Defects Found in component no.- {text_input}</h3>
                            <ul>
                                <li>No Defects Found</li>
                            </ul> """
            else: 
                html_list = f"""
                            <h3>Defects Found in component no.- {text_input}</h3>
                            <ul>
                                {''.join([f"<li>{v}</li>" for v in dl])}
                            </ul> """
            st.markdown(html_list, unsafe_allow_html=True)

            unique_defect = Counter(set(audit_list))

            # for i, upd_dict in enumerate(unique_defect, start=1):
            current_dict = initial_defect_count.copy()
            for key, value in unique_defect.items():
                if key in current_dict:
                    current_dict[key] += value
                
            row_values = [component_name] + list(current_dict.values())
            write_to_csv(row_values, "csv_audit/defect_report.csv")


            # print(row_values)
            # defect_df.loc[len(defect_df)] = row_values
            # defect_df.to_csv("csv_audit/defect_report.csv", header=True, index=False)
            defect_report = pd.read_csv("csv_audit/defect_report.csv")
            # defect_df.append(row_values, ignore_index=True)
            # defect_df = defect_df.append(row_values, ignore_index=True)
            st.markdown("""<h4>All Component Defects Report : </h4>""", unsafe_allow_html=True)
            st.dataframe(defect_report.set_index(defect_report.columns[0]))
            # st.markdown(defect_report.style.hide(axis="index").to_html(), unsafe_allow_html=True)
            
            # for key, value in unique_defect.items():
            #     if key in initial_defect_count:
            #         initial_defect_count[key] += value

            sum_values = defect_report.drop('Component', axis=1).sum()
            # print(sum_values.index)
            # print(sum_values.values)

            st.markdown("""<h4>Graphical View : </h4>""", unsafe_allow_html=True)

            color = ["#fd556f", "#db3d12", "#4b12db", "#1f2eb8", "#dbac12", "#12c0db", 
                    "#5b8c15", "#fd2ce5", "#8bd3c7", "#26bf4f", "#c42147", "#2163c4", 
                    "#5d13a8", "#12db91"] 

            
            fig1 = plt.figure(figsize=(20,5), facecolor="lightgray")
            plt.bar(sum_values.index, sum_values.values, color=color, width=0.8, align="center")
            ax = fig1.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            # fig1.align_xlabels()
            ax.set_facecolor("lightgray")
            ax.set_xticklabels(labels=sum_values.index,ha="right")
            # ax.bar()
            plt.xlabel("Defects", fontsize=12, fontweight="bold")
            plt.ylabel("Count", fontsize=12, fontweight="bold")
            plt.title("Counts of defect present in all component", fontsize=14, fontweight="bold")
            plt.xticks(rotation=70) #defect_report.columns[1:], 
            # ax.set_xticklabels(ha="center")
            st.pyplot(fig1)


            # defect_data = pd.DataFrame.from_dict(
            #     (k, v) for k, v in unique_defect.items()
            # )
            # defect_data.columns = ["Defects", "Count"]

            st.markdown("""<h4>Chart Report : </h4>""", unsafe_allow_html=True)
            # Custom HTML table with styled header
            html_table = f"""
                <table style="width: 100%;">
                    <thead>
                        <tr style="background-color: #3B6936; text-align: center; color: white; font-weight: bold; font-size: large;">
                            <th>Defects</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f"<tr><td>{d}</td><td>{c}</td><td>{round(c/text_input,4)*100}%</td></tr>" for d,c in zip(list(sum_values.index),list(sum_values.values))])}
                    </tbody>
                </table>
            """

            # Render the HTML table
            st.markdown(html_table, unsafe_allow_html=True)
            st.markdown(
                """<hr style="height:8px;border-radius:2px;color:#333;background-color:#3B6936;" /> """,
                unsafe_allow_html=True,
            )



    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
                with col1:
                    st.image(
                        uploaded_file,
                        caption=f"Image-{uploaded_file.name}",
                        use_column_width=True,
                    )

    # del uploaded_files


if __name__ == "__main__":
    main()
