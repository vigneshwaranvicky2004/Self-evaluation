import gradio as gr
import numpy as np
from sklearn.linear_model import Perceptron
import pandas as pd

file_path = "Student-Employability-Datasets (1).xlsx"
df = pd.read_excel(file_path, sheet_name='Data')

X = df.iloc[:, 1:-2].values  
y = (df['CLASS'] == 'Employable').astype(int) 

model = Perceptron()
model.fit(X, y)

def evaluate_employment(name, *ratings):
    input_data = np.array(ratings).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        return f"{name}, Congrats! 🎉 You are employable."
    else:
        return f"{name}, Try to upgrade yourself! 📚"

def app():
    with gr.Blocks() as demo:
        name = gr.Textbox(label="Enter your name")
        sliders = [gr.Slider(1, 5, step=1, label=col) for col in df.columns[1:-2]]
        button = gr.Button("Get Yourself Evaluated")
        output = gr.Textbox(label="Result")
        
        button.click(evaluate_employment, inputs=[name] + sliders, outputs=output)
    
    return demo

app().launch(share=True)

