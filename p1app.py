import streamlit as st
import PyPDF2
import docx
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import io
import base64

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load stop words
stop_words = set(stopwords.words('english'))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to clean and preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back to text
    return ' '.join(filtered_tokens)

# Function to extract skills from resume
def extract_skills(text):
    skills_db = [
        'python', 'java', 'c++', 'c#', 'javascript', 'typescript',
        'html', 'css', 'react', 'angular', 'vue', 'node.js',
        'django', 'flask', 'fastapi', 'spring', 'express',
        'sql', 'mysql', 'postgresql', 'mongodb', 'oracle',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes',
        'git', 'github', 'gitlab', 'jenkins', 'ci/cd',
        'agile', 'scrum', 'kanban', 'jira', 'confluence',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'data analysis', 'data science', 'machine learning', 'deep learning',
        'nlp', 'computer vision', 'artificial intelligence',
        'project management', 'leadership', 'communication',
        'problem solving', 'analytical thinking', 'critical thinking',
        'microsoft office', 'excel', 'powerpoint', 'word',
        'photoshop', 'illustrator', 'indesign', 'figma', 'sketch',
        'marketing', 'sales', 'customer service', 'hr', 'recruitment'
    ]
    
    skills_found = []
    for skill in skills_db:
        # Using word boundary to match whole words
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text.lower()):
            skills_found.append(skill)
    
    return skills_found

# Function to extract education from resume
def extract_education(text):
    education_keywords = [
        'bachelor', 'master', 'phd', 'doctor', 'degree',
        'b.s.', 'b.a.', 'm.s.', 'm.a.', 'ph.d.',
        'bs', 'ba', 'ms', 'ma', 'mba', 'mtech', 'btech',
        'university', 'college', 'institute', 'school',
        'graduated', 'graduation', 'graduate',
        'engineering', 'business', 'science', 'arts',
        'computer science', 'information technology', 'electronics',
        'mechanical', 'civil', 'electrical'
    ]
    
    education_info = []
    for keyword in education_keywords:
        pattern = r'(?i)(\b' + re.escape(keyword) + r'\b.{0,100})'
        matches = re.findall(pattern, text)
        education_info.extend(matches)
    
    # Remove duplicates and sort by length (longer entries likely have more details)
    education_info = list(set(education_info))
    education_info.sort(key=len, reverse=True)
    
    # Return top 3 education entries
    return education_info[:3]

# Function to extract experience from resume
def extract_experience(text):
    experience_patterns = [
        r'(?i)(\d+\+?\s*years?\s*(?:of)?\s*experience)',
        r'(?i)(experience\s*:\s*\d+\+?\s*years?)',
        r'(?i)(work\s*experience\s*:[^\n.]*)',
        r'(?i)(professional\s*experience\s*:[^\n.]*)',
        r'(?i)((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*-\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4})',
        r'(?i)((?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}\s*-\s*(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4})',
        r'(?i)((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*-\s*present)',
        r'(?i)((?:january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}\s*-\s*present)'
    ]
    
    experience_info = []
    for pattern in experience_patterns:
        matches = re.findall(pattern, text)
        experience_info.extend(matches)
    
    # Extract years of experience
    years_of_experience = 0
    years_pattern = r'(?i)(\d+)\+?\s*years?\s*(?:of)?\s*experience'
    years_match = re.search(years_pattern, text)
    if years_match:
        years_of_experience = int(years_match.group(1))
    
    # Return both details and years
    return {
        'details': list(set(experience_info)),
        'years': years_of_experience
    }

# Function to extract contact information from resume
def extract_contact_info(text):
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email = re.search(email_pattern, text)
    
    # Extract phone number
    phone_pattern = r'(?:\+\d{1,3}\s?)?(?:\(\d{3}\)\s?|\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}'
    phone = re.search(phone_pattern, text)
    
    # Extract LinkedIn URL
    linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+'
    linkedin = re.search(linkedin_pattern, text)
    
    return {
        'email': email.group(0) if email else None,
        'phone': phone.group(0) if phone else None,
        'linkedin': linkedin.group(0) if linkedin else None
    }

# Function to extract candidate name from resume
def extract_name(text):
    # This is a simple approach and may need refinement
    # Look for patterns like "Name: John Doe" or text at the beginning of the resume
    name_pattern = r'(?i)(?:name|full name|candidate)[\s:]+([A-Za-z\s]+)'
    name_match = re.search(name_pattern, text)
    
    if name_match:
        return name_match.group(1).strip()
    else:
        # Try to get the first line that might contain a name
        first_lines = text.split('\n')[:5]
        for line in first_lines:
            line = line.strip()
            # Check if line is short enough to be a name and doesn't contain common headers
            if 5 < len(line) < 50 and not any(word in line.lower() for word in ['resume', 'cv', 'curriculum', 'vitae', 'email', 'phone', 'address']):
                return line
    
    return "Unknown Candidate"

# Function to match resume with job requirements
def match_resume_with_job(resume_text, job_requirements):
    # Preprocess texts
    processed_resume = preprocess_text(resume_text)
    processed_job = preprocess_text(job_requirements)
    
    # Extract skills
    resume_skills = extract_skills(resume_text.lower())
    job_skills = extract_skills(job_requirements.lower())
    
    # Calculate skill match
    skill_matches = [skill for skill in resume_skills if skill in job_skills]
    skill_match_percentage = len(skill_matches) / len(job_skills) * 100 if job_skills else 0
    
    # Use TF-IDF for content similarity
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_resume, processed_job])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    except:
        similarity = 0
    
    # Calculate overall match score (weighted average)
    overall_match = (skill_match_percentage * 0.6) + (similarity * 0.4)
    
    return {
        'skill_match_percentage': skill_match_percentage,
        'content_similarity': similarity,
        'overall_match': overall_match,
        'matching_skills': skill_matches,
        'missing_skills': [skill for skill in job_skills if skill not in resume_skills]
    }

# Function to analyze resume strength and weaknesses
def analyze_resume_quality(resume_text):
    # Define metrics
    total_words = len(re.findall(r'\b\w+\b', resume_text))
    skills_count = len(extract_skills(resume_text))
    education_entries = extract_education(resume_text)
    experience_info = extract_experience(resume_text)
    contact_info = extract_contact_info(resume_text)
    
    # Basic analysis
    analysis = {
        'length': {
            'score': min(100, total_words / 10),
            'evaluation': 'Good' if total_words > 300 else 'Could be improved',
            'suggestion': 'Resume is concise' if total_words > 300 and total_words < 1000 else 
                         'Resume is too short' if total_words <= 300 else 'Resume may be too lengthy'
        },
        'skills': {
            'score': min(100, skills_count * 5),
            'evaluation': 'Good' if skills_count >= 10 else 'Could be improved',
            'suggestion': 'Good variety of skills mentioned' if skills_count >= 10 else 
                         'Consider adding more relevant skills'
        },
        'education': {
            'score': min(100, len(education_entries) * 33.33),
            'evaluation': 'Good' if education_entries else 'Missing',
            'suggestion': 'Education section is present' if education_entries else 
                         'Add education details to strengthen your resume'
        },
        'experience': {
            'score': min(100, experience_info['years'] * 20),
            'evaluation': 'Good' if experience_info['years'] >= 3 else 'Limited',
            'suggestion': 'Good experience demonstrated' if experience_info['years'] >= 3 else 
                         'Highlight your experience more effectively'
        },
        'contact': {
            'score': min(100, (bool(contact_info['email']) + bool(contact_info['phone']) + bool(contact_info['linkedin'])) * 33.33),
            'evaluation': 'Complete' if all(contact_info.values()) else 'Incomplete',
            'suggestion': 'Contact information is complete' if all(contact_info.values()) else 
                         'Add more contact details for better accessibility'
        }
    }
    
    # Calculate overall score
    overall_score = (
        analysis['length']['score'] * 0.1 +
        analysis['skills']['score'] * 0.3 +
        analysis['education']['score'] * 0.2 +
        analysis['experience']['score'] * 0.3 +
        analysis['contact']['score'] * 0.1
    )
    
    # Generate strengths and weaknesses
    strengths = []
    weaknesses = []
    
    for category, details in analysis.items():
        if details['score'] >= 70:
            strengths.append(f"{category.capitalize()}: {details['suggestion']}")
        else:
            weaknesses.append(f"{category.capitalize()}: {details['suggestion']}")
    
    return {
        'overall_score': overall_score,
        'category_scores': analysis,
        'strengths': strengths,
        'weaknesses': weaknesses
    }

# Function to generate a downloadable CSV report
def get_csv_download_link(df, filename="resume_analysis_report.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV Report</a>'
    return href

# Streamlit app
def main():
    st.title("HR Resume Analyzer")
    st.write("Upload multiple resumes and match them against job descriptions")
    
    # Add tabs for different functionalities
    tab1, tab2 = st.tabs(["Resume Analysis", "Job Matching"])
    
    with tab1:
        st.header("Resume Analysis")
        
        # Multiple file uploader
        uploaded_files = st.file_uploader("Upload Multiple Resumes", type=["pdf", "docx"], accept_multiple_files=True)
        
        if uploaded_files and st.button("Analyze Resumes"):
            # Process each resume
            st.subheader(f"Analyzing {len(uploaded_files)} Resumes")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Initialize results storage
            results = []
            
            # Process each file
            for i, file in enumerate(uploaded_files):
                try:
                    # Extract text from the uploaded file
                    if file.name.endswith('.pdf'):
                        resume_text = extract_text_from_pdf(file)
                    elif file.name.endswith('.docx'):
                        resume_text = extract_text_from_docx(file)
                    else:
                        st.error(f"Unsupported file format for {file.name}")
                        continue
                    
                    # Extract basic information
                    candidate_name = extract_name(resume_text)
                    skills = extract_skills(resume_text)
                    education = extract_education(resume_text)
                    experience = extract_experience(resume_text)
                    contact = extract_contact_info(resume_text)
                    
                    # Analyze resume quality
                    analysis = analyze_resume_quality(resume_text)
                    
                    # Add results to the list
                    results.append({
                        'file_name': file.name,
                        'candidate_name': candidate_name,
                        'skills': skills,
                        'education': education,
                        'experience': experience,
                        'contact': contact,
                        'analysis': analysis,
                        'resume_text': resume_text
                    })
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            # Display results table
            if results:
                st.subheader("Resume Analysis Results")
                
                # Create a dataframe for the results
                df = pd.DataFrame({
                    'Candidate': [r['candidate_name'] for r in results],
                    'File': [r['file_name'] for r in results],
                    'Skills Count': [len(r['skills']) for r in results],
                    'Experience (Years)': [r['experience']['years'] for r in results],
                    'Education Level': [len(r['education']) > 0 for r in results],
                    'Overall Score': [r['analysis']['overall_score'] for r in results]
                })
                
                # Sort by overall score
                df = df.sort_values('Overall Score', ascending=False)
                
                # Display the table
                st.dataframe(df)
                
                # Provide download link
                st.markdown(get_csv_download_link(df, "resume_analysis_summary.csv"), unsafe_allow_html=True)
                
                # Detailed view for each resume
                st.subheader("Detailed Analysis")
                for i, result in enumerate(results):
                    with st.expander(f"{result['candidate_name']} - {result['file_name']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Overall Score", f"{result['analysis']['overall_score']:.1f}%")
                        col2.metric("Skills Found", len(result['skills']))
                        col3.metric("Experience", f"{result['experience']['years']} years")
                        
                        # Skills
                        st.write("**Skills:**")
                        if result['skills']:
                            st.write(", ".join(result['skills']))
                        else:
                            st.write("No skills detected")
                        
                        # Education
                        st.write("**Education:**")
                        if result['education']:
                            for edu in result['education']:
                                st.write(f"• {edu}")
                        else:
                            st.write("No education details detected")
                        
                        # Experience
                        st.write("**Experience:**")
                        if result['experience']['details']:
                            for exp in result['experience']['details']:
                                st.write(f"• {exp}")
                        else:
                            st.write("No experience details detected")
                        
                        # Contact
                        st.write("**Contact:**")
                        if any(result['contact'].values()):
                            for key, value in result['contact'].items():
                                if value:
                                    st.write(f"• {key.capitalize()}: {value}")
                        else:
                            st.write("No contact information detected")
                        
                        # Strengths and weaknesses
                        st.write("**Strengths:**")
                        if result['analysis']['strengths']:
                            for strength in result['analysis']['strengths']:
                                st.write(f"✅ {strength}")
                        else:
                            st.write("No significant strengths detected")
                        
                        st.write("**Areas for Improvement:**")
                        if result['analysis']['weaknesses']:
                            for weakness in result['analysis']['weaknesses']:
                                st.write(f"❗ {weakness}")
                        else:
                            st.write("No significant weaknesses detected")
            
            else:
                st.warning("No resumes were successfully analyzed.")
    
    with tab2:
        st.header("Job Matching")
        
        # Job description input
        job_description = st.text_area("Enter Job Description:", height=150)
        
        # Multiple file uploader (reuse from tab1 if already uploaded)
        if 'uploaded_files' not in locals() or not uploaded_files:
            uploaded_files = st.file_uploader("Upload Resumes to Match", type=["pdf", "docx"], accept_multiple_files=True)
        
        if uploaded_files and job_description and st.button("Match Resumes with Job"):
            # Process each resume and match with job description
            st.subheader(f"Matching {len(uploaded_files)} Resumes to Job Description")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Initialize matching results
            matching_results = []
            
            # Process each file
            for i, file in enumerate(uploaded_files):
                try:
                    # Extract text from the uploaded file
                    if file.name.endswith('.pdf'):
                        resume_text = extract_text_from_pdf(file)
                    elif file.name.endswith('.docx'):
                        resume_text = extract_text_from_docx(file)
                    else:
                        st.error(f"Unsupported file format for {file.name}")
                        continue
                    
                    # Extract candidate name
                    candidate_name = extract_name(resume_text)
                    
                    # Match resume with job description
                    match_result = match_resume_with_job(resume_text, job_description)
                    
                    # Add to matching results
                    matching_results.append({
                        'file_name': file.name,
                        'candidate_name': candidate_name,
                        'match_result': match_result,
                        'resume_text': resume_text
                    })
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            # Display matching results
            if matching_results:
                st.subheader("Job Matching Results")
                
                # Create a dataframe for the results
                df = pd.DataFrame({
                    'Candidate': [r['candidate_name'] for r in matching_results],
                    'File': [r['file_name'] for r in matching_results],
                    'Overall Match (%)': [r['match_result']['overall_match'] for r in matching_results],
                    'Skill Match (%)': [r['match_result']['skill_match_percentage'] for r in matching_results],
                    'Content Similarity (%)': [r['match_result']['content_similarity'] for r in matching_results],
                    'Matching Skills': [len(r['match_result']['matching_skills']) for r in matching_results],
                    'Missing Skills': [len(r['match_result']['missing_skills']) for r in matching_results]
                })
                
                # Sort by overall match score
                df = df.sort_values('Overall Match (%)', ascending=False)
                
                # Round percentage columns
                percentage_columns = ['Overall Match (%)', 'Skill Match (%)', 'Content Similarity (%)']
                for col in percentage_columns:
                    df[col] = df[col].round(1)
                
                # Display the table
                st.dataframe(df)
                
                # Provide download link
                st.markdown(get_csv_download_link(df, "job_matching_results.csv"), unsafe_allow_html=True)
                
                # Detailed view for each resume
                st.subheader("Detailed Matching Analysis")
                for i, result in enumerate(matching_results):
                    with st.expander(f"{result['candidate_name']} - {result['file_name']}"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Overall Match", f"{result['match_result']['overall_match']:.1f}%")
                        col2.metric("Skill Match", f"{result['match_result']['skill_match_percentage']:.1f}%")
                        col3.metric("Content Similarity", f"{result['match_result']['content_similarity']:.1f}%")
                        
                        # Matching skills
                        st.write("**Matching Skills:**")
                        if result['match_result']['matching_skills']:
                            st.write(", ".join(result['match_result']['matching_skills']))
                        else:
                            st.write("No matching skills found")
                        
                        # Missing skills
                        st.write("**Missing Skills:**")
                        if result['match_result']['missing_skills']:
                            st.write(", ".join(result['match_result']['missing_skills']))
                        else:
                            st.write("No missing skills")
                        
                        # Recommendation
                        st.write("**Recommendation:**")
                        match_score = result['match_result']['overall_match']
                        if match_score >= 75:
                            st.write("✅ Strong match - Recommended for interview")
                        elif match_score >= 50:
                            st.write("🟨 Moderate match - Consider for interview")
                        else:
                            st.write("❌ Low match - Not recommended for this position")
            
            else:
                st.warning("No resumes were successfully matched with the job description.")

if __name__ == "__main__":
    main()