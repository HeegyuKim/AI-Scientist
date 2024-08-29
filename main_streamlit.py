import streamlit as st
import pandas as pd
from openai import OpenAI
from ai_scientist.idea_gen_simple import generate_ideas, assess_idea, check_idea_novelty
from ai_scientist.perform_review import load_paper, perform_review
import json
import uuid, os

EXAMPLE_TASK = "You are given the following file to work with, which studies the phenomenon of grokking in neural networks by training multiple small Transformer models on multiple datasets of mathematical operations. The abstract for the original paper is \"In this paper we propose to study generalization of neural networks on small algorithmically generated datasets. In this setting, questions about data efficiency, memorization, generalization, and speed of learning can be studied in great detail. In some situations we show that neural networks learn through a process of 'grokking' a pattern in the data, improving generalization performance from random chance level to perfect generalization, and that this improvement in generalization can happen well past the point of overfitting. We also study generalization as a function of dataset size and find that smaller datasets require increasing amounts of optimization for generalization. We argue that these datasets provide a fertile ground for studying a poorly understood aspect of deep learning: generalization of overparametrized neural networks beyond memorization of the finite training dataset.\" Please come up with interesting experiments to investigate this phenomenon."
EXAMPLE_SEED_IDEAS = pd.DataFrame([
  {
    "Name": "adaptive_block_size",
    "Title": "Adaptive Block Size: Dynamic Context Window Adjustment for Efficient Training",
    "Experiment": "Modify the model to dynamically adjust its block size during training, starting with a smaller block size and gradually increasing it. This could potentially lead to faster initial training and better long-range dependency learning.",
    "Interestingness": 6,
    "Feasibility": 4,
    "Novelty": 4
  },
  {
    "Name": "layerwise_learning_rates",
    "Title": "Layer-wise Learning Rate Adaptation: Optimizing Training Dynamics in Transformer Models",
    "Experiment": "Implement layer-wise learning rates, where each transformer layer has its own learning rate. Modify the configure_optimizers function to assign different learning rates to different layers, with deeper layers having lower learning rates. Compare the training dynamics, convergence speed, and final performance with the baseline model.",
    "Interestingness": 4,
    "Feasibility": 6,
    "Novelty": 2
  }
])

client = OpenAI()


st.title('iKnow Lab AI Scientist')

models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]
pages = ["Idea Generation", "Idea Assessment", "AI Review"]

with st.sidebar:
    st.write('## Sidebar')
    model = st.selectbox('Select a model', models)

def idea_generation():
    st.write('## Idea Generation')

    task_description = st.text_area('Task Description', EXAMPLE_TASK, key='idea_gen_task_description')
    st.info("NOTE: CSV file must contain columns: Name, Title, Experiment, Interestingness, Feasibility, Novelty")
    csv_file = st.file_uploader('Upload CSV File', type=['csv'])
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        if df.columns[0] == 'Unnamed: 0':
            df = df.drop(columns=['Unnamed: 0'])
        
        columns=['Name', 'Title', 'Experiment', 'Interestingness', 'Feasibility', 'Novelty']
        # check columns
        if not all([col in df.columns for col in columns]):
            st.error(f'Columns must be {columns}')
            st.stop()
        edited_df = st.data_editor(df, num_rows="dynamic")
    else:
        edited_df = st.data_editor(EXAMPLE_SEED_IDEAS, num_rows="dynamic")


    num_reflection = st.number_input('Number of Reflections', min_value=1, max_value=10, value=3, key='idea-gen-num_reflection')
    max_ideas = st.number_input('Number of Generations', min_value=1, max_value=10, value=1, key='idea-gen-max_ideas')
    novelty_check = st.number_input('Number of Novelty Check Iterations', min_value=0, max_value=10, value=0, key='idea-gen-novelty_check', help='Number of iterations to check novelty (0 to disable)')

    generate = st.button('Generate Idea')

    if generate:
        with st.spinner('Generating Ideas...'):
            ideas = [json.dumps(row) for row in edited_df.to_dict('records')]
            ideas = generate_ideas(client, model, task_description, ideas, num_reflection, max_ideas)

        if novelty_check > 0:
            with st.spinner('Checking Novelty...'):
                ideas = check_idea_novelty(ideas, client, model, task_description, max_num_iterations=novelty_check)

        df = pd.DataFrame(ideas)
        st.dataframe(df)
        st.json(ideas)


def idea_assessment():
    st.write('## Idea Assessment')
    task_description = st.text_area('Task Description', EXAMPLE_TASK)
    title = st.text_input('Title', EXAMPLE_SEED_IDEAS.iloc[0]['Title'])
    experiment = st.text_area('Experiment', EXAMPLE_SEED_IDEAS.iloc[0]['Experiment'])

    generate = st.button('Assess Idea')
    if generate:
        with st.spinner('Assessing Idea...'):
            text, assessment = assess_idea(client, model, task_description, title, experiment)
        st.write('## Assessment')
        st.markdown(text)
        st.json(assessment)


def ai_review():
    st.write('## AI Review')

    pdf_file = st.file_uploader('Upload PDF File', type=['pdf'])
    max_pages = st.number_input('Max Pages', min_value=1, max_value=100, value=9)
    num_reflections = st.number_input('Number of Reflections', min_value=1, max_value=10, value=3)
    num_reviews_ensemble = st.number_input('Number of Reviews Ensemble', min_value=1, max_value=10, value=3)

    do_review = st.button('Generate Review')
    if pdf_file is not None and do_review:
        # save temp file
        temp_filename = f"{uuid.uuid4()}.pdf"
        with open(temp_filename, 'wb') as tf:
            tf.write(pdf_file.read())
            pdf_file = tf.name
            # Load the paper
            with st.spinner('Loading Paper...'):
                paper = load_paper(pdf_file, max_pages)
        os.remove(temp_filename)

        with st.spinner('Performing Review...'):
            review = perform_review(
                paper, model, client,
                num_reflections=num_reflections,
                num_reviews_ensemble=num_reviews_ensemble,
                )
        st.subheader(f"{model.capitalize()} Review")
        # st.json(review)
        for key, value in review.items():
            if isinstance(value, int):
                st.write(f"### {key}")
            else:
                st.write(f"## {key}")

            if isinstance(value, list):
                for v in value:
                    st.markdown(v)
            else:
                st.markdown(value)
            
        st.json(review)



for page, tab in zip(pages, st.tabs(pages)):
    with tab:
        if page == 'Idea Generation':
            idea_generation()
        elif page == 'Idea Assessment':
            idea_assessment()
        elif page == 'AI Review':
            ai_review()
