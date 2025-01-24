
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:11:03 2025

@author: ha
"""
#%%
import os
import PyPDF2
import pandas as pd
from dotenv import load_dotenv
#import streamlit as st
import re
from openai import OpenAI
#import sys
#import pysqlite3
#sys.modules['sqlite3'] = pysqlite3
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI

import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("API_KEY")
os.environ["OPENAI_API_KEY"] = api_key  # Add this after getting api_key


# Initialize OpenAI client
client = OpenAI(api_key=api_key)

#llm = LLM(model="openai/gpt-4o")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

#%%

# Reading PDF function
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

#%%
def process_resume(resume_text, job_description_text):
    # Create the agents
    job_requirements_agent = Agent(
        role="Job Requirements Extractor",
        goal="Extract key skills, qualifications, and experiences required for the job.",
        verbose=False,
        allow_delegation=False,
        backstory=(
            "You are a diligent Job Requirements Extractor. Your sole responsibility is to read the provided "
            "job description and extract the essential requirements."
        )
    )

    resume_analyzer_agent = Agent(
        role="Resume Analyzer",
        goal="Analyze the provided resume to identify the candidate's skills, qualifications, and experiences.",
        verbose=False,
        allow_delegation=False,
        backstory=(
            "You are provided with a resume. Your task is to scrutinize this text to understand the candidate's "
            "background and capabilities."
        )
    )

    resume_scorer_agent = Agent(
        role="Resume Scorer",
        goal="Score each resume based on how well it matches the job requirements.",
        verbose=False,
        allow_delegation=False,
        backstory=(
        "As a Resume Scorer, you evaluate each candidate's fit by comparing their "
        "analyzed resume against the extracted job requirements. Always begin your "
        "response with 'The score is x out of 10', replacing 'x' with the actual " 
        "score. Then provide a concise justification for the score - not too long "
        "or too short."
        )
    )


    # Create the new agents
    name_extractor_agent = Agent(
        role="Name Extractor",
        goal="Extract the candidate's name from the provided resume text.",
        verbose=False,
        allow_delegation=False,
        backstory=(
        "As a Resume Scorer, you evaluate each candidate's fit by comparing their "
        "analyzed resume against the extracted job requirements. Always begin your "
        "response with 'The score is x out of 10', replacing 'x' with the actual "
        "score. Then provide a justification for the score. Do not delegate this "
        "task; do it yourself."
        )
    )


    #############################
    # Define tasks
    job_requirements_task = Task(
        description=(
            "Extract key skills, qualifications, and experiences from the provided ({job_description_text})."
        ),
        expected_output=(
            "A structured list of job requirements, including necessary skills, qualifications, and experiences."
        ),
        agent=job_requirements_agent,
        async_execution=True,
    )

    resume_analysis_task = Task(
        description=(
            "Analyze the resume from ({resume_text}) to identify the candidate's skills, qualifications, "
            "and experiences."
        ),
        expected_output=(
            "A detailed profile of the candidate's skills, qualifications, and experiences."
        ),
        agent=resume_analyzer_agent,
        async_execution=True,
        inputs={'resume_text': resume_text}
    )

    resume_scoring_task = Task(
        description=(
            "Compare the candidate's analyzed profile from resume_analysis_task with the job requirements "
            "extracted from the job_requirements_task "
            "and score the resume on a scale of 1 to 10, with 1 being a very weak candidate and 10 being a perfect fit. Provide a justification for the score."
            "if you found out that the candidate is overqualified, it should affect your evaluation negatively and lower the score."
        ),
        expected_output=(
        "The score is [1-10] out of 10. [Justification for the score.]"
        ),
        context=[job_requirements_task, resume_analysis_task],
        agent=resume_scorer_agent
    )

    # Define the tasks
    name_extraction_task = Task(
        description=(
            "Extract the candidate's name from the provided resume text ({resume_text})."
        ),
        expected_output=(
            "The candidate's full name as it appears in the resume."
        ),
        agent=name_extractor_agent,
        #async_execution=True,
        inputs={'resume_text': resume_text}
    )
    # Provide inputs
    crew_inputs = {'job_description_text': job_description_text, 'resume_text': resume_text}

    # Create the crew
    talent_development_crew = Crew(
    agents=[job_requirements_agent, resume_analyzer_agent, resume_scorer_agent,name_extractor_agent],
    tasks=[job_requirements_task, resume_analysis_task, name_extraction_task, resume_scoring_task],
    manager = llm, verbose = True)
    #llm = llm, verbose=True)
    # Run the crew
    result = talent_development_crew.kickoff(inputs=crew_inputs)
    # Extract score and justification using regex

    regex = re.compile(r'The score is (\d+) out of 10\.\s*(.*)')
    match = regex.search(str(result))  # Convert result to string
    #match = regex.searchs(result.raw)

    if match:
        score = int(match.group(1))
        justification = match.group(2).strip()
    else:
        score = None
        justification = None


    # Extract name from name_extraction_task output
    #name = name_extraction_task.output.raw
    name = str(name_extraction_task.output.raw)  # Convert output to string
    return name, score, justification


#%%

def main():
    st.title("Resume Scoring Application")
    st.write("Upload resumes and provide a job description to score candidates based on their fit for the job.")

    # Job Description Input
    st.header("Job Description")
    job_description_text = st.text_area("Enter the job description here:", height=300)

    # Resume Upload
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])

    if st.button("Process Resumes"):
        if not job_description_text:
            st.error("Please provide a job description.")
        elif not uploaded_files:
            st.error("Please upload at least one resume.")
        else:
            results = []

            # Process each uploaded resume
            for uploaded_file in uploaded_files:
                resume_text = read_pdf(uploaded_file)
                # Process the resume
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    name, score, justification = process_resume(resume_text, job_description_text)

                # Display the result
                st.subheader(f"Results for ({uploaded_file.name})")
                st.write(f"**Score:** {score} out of 10")
                st.write(f"**Justification:** {justification}")

                # Append to results
                results.append({
                    'filename': uploaded_file.name,
                    'applicant_name': name,
                    'score': score,
                    'justification': justification
                })

            # Create a DataFrame
            df_results = pd.DataFrame(results)

            # Optionally, display the DataFrame
            st.header("Summary")
            st.dataframe(df_results)

            # Optionally, allow download of results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='resume_scoring_results.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
