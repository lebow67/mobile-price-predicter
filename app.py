import gradio as gr
import pandas as pd
import joblib

model = joblib.load("mobile_price_model.pkl")
expected_columns = joblib.load("mobile_price_columns.pkl")

PRICE_LABELS = {
    0: "💚 Low Cost",
    1: "💛 Medium Cost",
    2: "🧡 High Cost",
    3: "❤️ Very High Cost"
}

def predict_price(battery_power, blue, clock_speed, dual_sim, fc, four_g,
                   int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                   px_width, ram, sc_h, sc_w, talk_time, three_g,
                   touch_screen, wifi):

    row = {
        "battery_power": battery_power,
        "blue": int(blue),
        "clock_speed": clock_speed,
        "dual_sim": int(dual_sim),
        "fc": fc,
        "four_g": int(four_g),
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": int(three_g),
        "touch_screen": int(touch_screen),
        "wifi": int(wifi)
    }

    input_df = pd.DataFrame([row])
    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]
    return PRICE_LABELS[prediction]


with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    gr.Markdown("# 📱 Mobile Price Range Predictor")
    gr.Markdown("Adjust the specifications below to predict the phone's price category.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Core Specs")
            ram = gr.Slider(256, 4000, value=2000, step=1, label="RAM (MB)")
            battery_power = gr.Slider(500, 2000, value=1200, step=1, label="Battery Power (mAh)")
            clock_speed = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Clock Speed (GHz)")
            n_cores = gr.Slider(1, 8, value=4, step=1, label="Number of Cores")
            int_memory = gr.Slider(2, 64, value=32, step=1, label="Internal Memory (GB)")

        with gr.Column():
            gr.Markdown("### Display & Camera")
            px_height = gr.Slider(0, 2000, value=800, step=1, label="Pixel Resolution Height")
            px_width = gr.Slider(500, 2000, value=1200, step=1, label="Pixel Resolution Width")
            sc_h = gr.Slider(5, 20, value=12, step=1, label="Screen Height (cm)")
            sc_w = gr.Slider(0, 18, value=7, step=1, label="Screen Width (cm)")
            pc = gr.Slider(0, 20, value=10, step=1, label="Primary Camera (MP)")
            fc = gr.Slider(0, 20, value=5, step=1, label="Front Camera (MP)")

        with gr.Column():
            gr.Markdown("### Other Specs")
            mobile_wt = gr.Slider(80, 200, value=140, step=1, label="Mobile Weight (g)")
            m_dep = gr.Slider(0.1, 1.0, value=0.5, step=0.1, label="Mobile Depth (cm)")
            talk_time = gr.Slider(2, 20, value=10, step=1, label="Talk Time (hrs)")

            gr.Markdown("### Features")
            with gr.Row():
                blue = gr.Checkbox(label="Bluetooth")
                dual_sim = gr.Checkbox(label="Dual SIM")
                wifi = gr.Checkbox(label="WiFi")
            with gr.Row():
                four_g = gr.Checkbox(label="4G")
                three_g = gr.Checkbox(label="3G")
                touch_screen = gr.Checkbox(label="Touch Screen")

    predict_btn = gr.Button("🔮 Predict Price Range", variant="primary")
    output = gr.Textbox(label="Predicted Price Category", scale=2)

    predict_btn.click(
        fn=predict_price,
        inputs=[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                px_width, ram, sc_h, sc_w, talk_time, three_g,
                touch_screen, wifi],
        outputs=output
    )

demo.launch()