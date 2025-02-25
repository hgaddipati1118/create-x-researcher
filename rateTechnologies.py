import csv
import random  # Add this import for random sampling
from pydantic import BaseModel
from askLlama import ask_llm, ask_llm_with_schema
import markdown2  # Make sure to pip install markdown2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os
import subprocess
from pathlib import Path

# Define the LLM model to use throughout the application
LLM_MODEL = "qwen2.5:3b"

# New imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Add markdown support with markdown2 library
import markdown2
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Flowable, Paragraph
from io import BytesIO
from xhtml2pdf import pisa
import html

# Define a Pydantic model for collecting the rating from the LLM.
class RatingResponse(BaseModel):
    feasibility: int
    effectiveness: int
    cost: int
    innovation: int
    scalability: int
    relevance: int
    impact: int

class AnalysisResponse(BaseModel):
    analysis: str

class CleanedTargetProblem(BaseModel):
    cleaned_target_problem: str
    confidence_score: int

class PDFMarkdownConverter:
    """
    A utility class to convert markdown to PDF content compatible with ReportLab.
    """
    # Default style names - can be changed by calling code
    heading1_style = 'Heading1'
    heading2_style = 'Heading2'
    heading3_style = 'Heading3'
    
    @staticmethod
    def convert_markdown_to_html(markdown_text):
        """Convert markdown text to HTML"""
        html_content = markdown2.markdown(markdown_text, extras=["tables", "code-friendly"])
        return html_content
    
    @staticmethod
    def create_paragraph_from_markdown(markdown_text, styles):
        """Convert markdown to a ReportLab paragraph"""
        html_content = PDFMarkdownConverter.convert_markdown_to_html(markdown_text)
        # Replace markdown headers with styled paragraphs using the configured style names
        html_content = html_content.replace('<h1>', f'<para style="{PDFMarkdownConverter.heading1_style}">').replace('</h1>', '</para>')
        html_content = html_content.replace('<h2>', f'<para style="{PDFMarkdownConverter.heading2_style}">').replace('</h2>', '</para>')
        html_content = html_content.replace('<h3>', f'<para style="{PDFMarkdownConverter.heading3_style}">').replace('</h3>', '</para>')
        
        # Handle other formatting
        html_content = html_content.replace('<strong>', '<b>').replace('</strong>', '</b>')
        html_content = html_content.replace('<em>', '<i>').replace('</em>', '</i>')
        
        return Paragraph(html_content, styles['Normal'])

def evaluate_technology(row, target_problem, potential_solution):
    """
    Construct a prompt from the technology details and the user-supplied target problem.
    Call ask_llm_with_schema to get ratings (1-100) from the LLM based on how well
    the provided solution addresses the target problem for each aspect: feasibility, effectiveness, cost, and innovation.
    """
    tech_name = row.get("Technology Name", "Unknown Technology")
    solution = row.get("Solution", "No solution provided")
    
    question = (
        f"Consider the problem: '{target_problem}'. "
        f"Consider the following solution: '{solution}'. "
        f"Here is a way the solution could help solve the problem: {potential_solution}. "
        "Rate each aspect on a 1-100 scale using these guidelines:\n\n"
        "Feasibility (Implementation Potential):\n"
        "81-100: Requires minimal resources/time, uses proven tools\n"
        "41-60: Needs moderate adjustments (e.g., new team skills)\n"
        "1-20: Unrealistic (needs unproven tech)\n\n"
        "Effectiveness (Problem Solving):\n"
        "81-100: Validated with strong metrics (e.g., 95% accuracy)\n"
        "41-60: Partial success with testing gaps\n"
        "1-20: No evidence of solving problem\n\n"
        "Cost Efficiency:\n"
        "81-100: Highly cost-efficient (open-source tools)\n"
        "41-60: Moderate budget with justifiable ROI\n"
        "1-20: Prohibitively expensive\n\n"
        "Innovation:\n"
        "81-100: Breakthrough idea (new algorithm)\n"
        "41-60: Iterative improvement\n"
        "1-20: No originality\n\n"
        "Scalability:\n"
        "81-100: Works across contexts with minimal tweaks\n"
        "41-60: Limited to specific use cases\n"
        "1-20: Fails under demand\n\n"
        "Relevance:\n"
        "81-100: Directly targets core requirements\n"
        "41-60: Partially addresses prompt\n"
        "1-20: Off-topic/misaligned\n\n"
        "Impact:\n"
        "81-100: Transforms field, addresses risks\n"
        "41-60: Minor benefits with trade-offs\n"
        "1-20: Harmful/ignores consequences\n\n"
        "IMPORTANT: Your response MUST include ALL seven ratings. Return JSON with integer ratings (1-100) for ALL of these fields: "
        "feasibility, effectiveness, cost, innovation, scalability, relevance, impact.\n"
        "Example of correct format: {\"feasibility\": 75, \"effectiveness\": 80, \"cost\": 60, \"innovation\": 70, \"scalability\": 65, \"relevance\": 85, \"impact\": 75}"
    )
    
    # First try with the Pydantic schema
    try:
        rating_response = ask_llm_with_schema(question, RatingResponse, model=LLM_MODEL, temperature=0)
        print(f"Received valid rating response: {rating_response}")
        return rating_response
    except Exception as e:
        print(f"Error with schema validation: {e}")
        
        # Try to get a raw response and parse it manually
        try:
            # Use the regular ask_llm function instead
            raw_response = ask_llm(question, model=LLM_MODEL, temperature=0)
            print(f"Got raw response: {raw_response}")
            
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON-like structure in the response
            json_match = re.search(r'\{.*?\}', raw_response.replace('\n', ' '), re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Create a RatingResponse with default values for missing fields
                default_fields = {
                    "feasibility": 50, 
                    "effectiveness": 50, 
                    "cost": 50,
                    "innovation": 50, 
                    "scalability": 50, 
                    "relevance": 50,
                    "impact": 50
                }
                
                # Update with any values from the response
                for field in default_fields.keys():
                    if field in data:
                        default_fields[field] = data[field]
                
                print(f"Manually extracted fields: {default_fields}")
                return RatingResponse(**default_fields)
            else:
                raise ValueError("No JSON structure found in response")
                
        except Exception as inner_error:
            print(f"Failed to manually parse response: {inner_error}")
            
        # Return default values as a last resort
        print(f"Using default rating for '{tech_name}'")
        return RatingResponse(
            feasibility=50, effectiveness=50, cost=50, innovation=50,
            scalability=50, relevance=50, impact=50
        )

def load_technologies(csv_file_path: str):
    """
    Load technology records from the provided CSV file.
    """
    technologies = []
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                technologies.append(row)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
    return technologies

def automatic_all(techs):
    """
    Automatically process all technology records.
    The user is first prompted for a target problem and filtering parameters,
    then each technology's solution is evaluated for its rating.
    """
    target_problem = input("Enter the target problem you want solutions to address: ").strip()
    target_problem = clean_target_problem(target_problem)
    
    # Get filtering parameters upfront
    filter_params = get_filtering_parameters()
    print(f"\nWill generate report based on your filter: {filter_params['type']} = {filter_params['value']}")
    
    for index, row in enumerate(techs, start=1):
        print(f"\nEvaluating Technology {index}/{len(techs)}:")
        
        potential_solutions = []
        for i in range(5):
            print(f"Generating potential solution {i+1} of 5:")
            potential_solutions.append(generate_solution(row, target_problem, potential_solutions))
        row['potential_solutions'] = potential_solutions

        max_rating = RatingResponse(
            feasibility=0, effectiveness=0, cost=0, innovation=0,
            scalability=0, relevance=0, impact=0
        )
        max_total_rating = 0
        potential_ratings = []
        
        for i in range(5):
            print(f"Evaluating potential solution {i+1} of 5:")
            try:
                rating = evaluate_technology(row, target_problem, potential_solutions[i])
                
                # Verify we got a valid RatingResponse object
                if not isinstance(rating, RatingResponse):
                    print(f"Warning: Invalid rating type received. Using default rating.")
                    rating = RatingResponse(
                        feasibility=0, effectiveness=0, cost=0, innovation=0,
                        scalability=0, relevance=0, impact=0
                    )
                
                # Calculate total rating
                total_rating = sum([
                    rating.feasibility, rating.effectiveness, rating.cost,
                    rating.innovation, rating.scalability, rating.relevance,
                    rating.impact
                ]) // 7
                
                potential_ratings.append(rating)
                if total_rating > max_total_rating:
                    max_total_rating = total_rating
                    max_rating = rating
            except Exception as e:
                print(f"Error evaluating solution {i+1}: {e}")
                # Create a default rating for this solution
                default_rating = RatingResponse(
                    feasibility=0, effectiveness=0, cost=0, innovation=0,
                    scalability=0, relevance=0, impact=0
                )
                potential_ratings.append(default_rating)
        
        row['rating'] = max_rating
        row['potential_ratings'] = potential_ratings
    
    # After processing, sort technologies
    try:
        rated_techs = sorted(techs, key=lambda x: calculate_total_rating(x['rating']), reverse=True)
        
        # Filter technologies based on parameters selected earlier
        filtered_techs, report_title = filter_technologies(rated_techs, filter_params)
        
        if not filtered_techs:
            print("\nNo technologies met the filtering criteria.")
            return
            
        print(f"\n{report_title} Based on LLM Rating:")
        for idx, tech in enumerate(filtered_techs, start=1):
            tech_name = tech.get("Technology Name", "Unknown")
            total_rating = calculate_total_rating(tech['rating'])
            print(f"{idx}. {tech_name} (Rating: {total_rating})")
            
        # Create report with filtered technologies
        create_pdf_report(filtered_techs, target_problem, report_title)
        
    except Exception as e:
        print(f"Error sorting technologies: {e}")
        print("Attempting to continue with unsorted technologies...")
        top5 = techs[:5] if len(techs) >= 5 else techs
        create_pdf_report(top5, target_problem, "Technology Report (Unsorted)")

def calculate_total_rating(rating):
    """
    Calculate the total rating score from a RatingResponse object.
    Returns the average of all rating components.
    """
    if not rating:
        return 0
    
    try:
        total = sum([
            rating.feasibility, 
            rating.effectiveness, 
            rating.cost,
            rating.innovation, 
            rating.scalability, 
            rating.relevance,
            rating.impact
        ])
        return total // 7  # Integer division for a whole number result
    except (AttributeError, TypeError):
        return 0

def create_pdf_report(top_techs, target_problem, report_title="Top Technologies Analysis"):
    """
    Create a report by first generating a markdown file, then converting it to PDF using mdpdf.
    All reports are stored in a 'reports' folder with a concise LLM-generated title.
    """
    # Generate the markdown file in the reports folder
    md_file = create_markdown_report(top_techs, target_problem, report_title, report_folder="reports")
    
    # Convert the markdown file to PDF
    pdf_file = convert_markdown_to_pdf(md_file, report_folder="reports")
    
    if pdf_file:
        print(f"Report has been created: {pdf_file}")
    else:
        print("PDF conversion not available. You can view the markdown file directly:")
        print(f"  {md_file}")
        
    return md_file  # Always return the markdown file path

def getSimplifiedExplanation(potential_solution, target_problem):
    """
    Generate a simplified explanation of the solution that would be understandable
    to a high school student.
    """
    prompt = (
        f"Consider the following potential solution to a problem: '{potential_solution}'. "
        f"The problem being addressed is: '{target_problem}'. "
        "Please explain this solution in simple terms that a high school student would understand. "
        "Avoid technical jargon, use simple analogies where helpful, and explain the core concepts in about 3-4 paragraphs. "
        "Format your response in markdown with a conversational tone."
    )
    return ask_llm(prompt, model=LLM_MODEL)

def getSolutionAnalysis(potential_solution, rating, target_problem):
    """
    Generate an analysis of the provided solution in markdown format.
    Avoids repeating the rating categories in the analysis.
    """
    prompt = (
        f"Consider the following potential solution: '{potential_solution}'. "
        f"Given the target problem: '{target_problem}', please explain how the rating of the solution is: {rating}. "
        f"The rating is on a scale of 1-100, where 1 is the worst and 100 is the best. And also the rating is based on the following criteria: feasibility, effectiveness, cost, innovation, scalability, relevance, impact. "
        "Provide a detailed analysis of how this solution addresses the problem."
    )
    return ask_llm(prompt, model=LLM_MODEL)

def getNextSteps(potential_solution, target_problem):
    """
    Generate next steps for implementing the provided solution in markdown format.
    """
    prompt = (
        f"Consider the following potential solution: '{potential_solution}'. "
        f"Given the target problem: '{target_problem}', please suggest how to implement the solution and give the next steps to take considering that someone is just starting on this solution. Have it be under 2000 tokens and be concise. "
        "Format your response in markdown with a '## Implementation Steps' heading followed by numbered steps. Use bullet points for details under each step, and bold text for important points."
    )
    return ask_llm(prompt, model=LLM_MODEL)

def getSolutionEffectiveness(potential_solution, target_problem):
    """
    Generate an explanation of why this specific solution is effective for the target problem,
    focusing on the match between solution capabilities and problem requirements.
    """
    prompt = (
        f"Consider the following potential solution: '{potential_solution}' "
        f"and the target problem: '{target_problem}'. "
        "Explain specifically why this solution is well-suited to address this particular problem. "
        "Focus on how the solution's specific features directly address key aspects of the problem. "
        "Include concrete examples of how the solution would work in practice for this problem. "
        "Format your response in markdown with a clear structure highlighting the key connections "
        "between the solution and problem. Keep the response between 150-300 words."
    )
    return ask_llm(prompt, model=LLM_MODEL)

def generate_solution(tech, target_problem, potential_solutions):
    """
    Generate a distinct way the technology's solution could help solve the target problem.
    """
    tech_name = tech.get("Technology Name", "Unknown Technology")
    solution = tech.get("Solution", "No solution provided")
    prompt = (
        f"Technology '{tech_name}' proposes the solution: '{solution}'. "
        f"Given the target problem: '{target_problem}', please brainstorm a distinct way in "
        "which this solution could help solve the problem that is not already in the list of potential solutions. "
        f"The list of potential solutions is: {','.join(potential_solutions)}."
        f" Keep it short and concise just one idea have it be under 100 tokens."
    )
    try:
        return ask_llm(prompt, model=LLM_MODEL)
    except Exception as e:
        print(f"Error processing analysis for technology '{tech_name}': {e}")
        return "No analysis available."

def clean_target_problem(target_problem):
    """
    Clean the target problem by interactively working with the user to refine it
    until it's suitable for LLM processing.
    """
    cleaned_target_problem = target_problem.strip()
    
    while True:
        prompt = (
            f"Given the target problem: '{cleaned_target_problem}', analyze and improve it following these steps:\n\n"
            "1. Paraphrase the Core Objective\n"
            "   - Simplify jargon or ambiguous terms\n"
            "2. Identify Missing Information\n"
            "   - Highlight assumptions or gaps in the description\n"
            "3. Break Down Complexity\n"
            "   - Split multi-part problems into focused sub-questions\n"
            "4. Add Constraints or Context\n"
            "   - Specify scope, resources, or measurable goals if absent\n"
            "5. Align With LLM Strengths\n"
            "   - Frame the problem to leverage structured planning and reasoning\n\n"
            "Rate your confidence in the problem statement's quality from 1-100. "
            "Return a JSON with the structure: {\"cleaned_target_problem\": \"<improved_problem_statement>\", \"confidence_score\": <score>}."
        )
        
        response = ask_llm_with_schema(prompt, CleanedTargetProblem, model=LLM_MODEL, temperature=0)
        
        if response.confidence_score >= 75:
            print(f"\nFinal problem statement: {response.cleaned_target_problem}")
            print(f"Confidence score: {response.confidence_score}/100")
            return response.cleaned_target_problem
        
        print(f"\nCurrent problem statement: {response.cleaned_target_problem}")
        print(f"Clarity score: {response.confidence_score}/100 (Target: 75+)")
        print("The problem statement could use more clarity or specificity.")
        
        user_input = input("Would you like to: \n1. Accept this statement\n2. Modify it yourself\n3. Get suggestions\nEnter choice (1/2/3): ")
        
        if user_input == "1":
            return response.cleaned_target_problem
        elif user_input == "2":
            new_statement = input("Enter your revised problem statement: ")
            cleaned_target_problem = new_statement.strip()
        elif user_input == "3":
            suggestion_prompt = (
                f"The current problem statement is: '{response.cleaned_target_problem}'\n\n"
                "Provide specific suggestions to improve this problem statement by:\n"
                "1. Identifying any vague terms that need clarification\n"
                "2. Suggesting missing context or constraints\n"
                "3. Recommending how to make it more specific and actionable\n"
                "4. Giving a concrete example of a revised version\n"
            )
            suggestions = ask_llm(suggestion_prompt, model=LLM_MODEL)
            print(f"\nSuggestions for improvement:\n{suggestions}")
            new_statement = input("\nBased on these suggestions, enter a revised problem statement (or press Enter to keep current): ")
            if new_statement.strip():
                cleaned_target_problem = new_statement.strip()
        else:
            print("Invalid choice. Please try again.")

def random_sample_technologies(techs):
    """
    Randomly sample a subset of technologies for evaluation.
    The user specifies how many technologies to sample and filtering parameters upfront.
    """
    if not techs:
        print("No technologies available to sample from.")
        return
    
    total_techs = len(techs)
    print(f"\nYou have {total_techs} technologies available.")
    
    while True:
        try:
            sample_size = input(f"How many technologies would you like to randomly sample? (1-{total_techs}): ").strip()
            sample_size = int(sample_size)
            
            if 1 <= sample_size <= total_techs:
                break
            else:
                print(f"Please enter a number between 1 and {total_techs}.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Randomly sample the specified number of technologies
    sampled_techs = random.sample(techs, sample_size)
    print(f"Randomly sampled {sample_size} technologies for evaluation.")
    
    # Get filtering parameters upfront
    filter_params = get_filtering_parameters()
    print(f"\nWill generate report based on your filter: {filter_params['type']} = {filter_params['value']}")
    
    # Use the same process as automatic_all but with the sampled technologies
    target_problem = input("Enter the target problem you want solutions to address: ").strip()
    target_problem = clean_target_problem(target_problem)
    
    for index, row in enumerate(sampled_techs, start=1):
        print(f"\nEvaluating Technology {index} of {sample_size}:")
        
        potential_solutions = []
        for i in range(5):
            print(f"Generating potential solution {i+1} of 5:")
            potential_solutions.append(generate_solution(row, target_problem, potential_solutions))
        row['potential_solutions'] = potential_solutions

        max_rating = RatingResponse(
            feasibility=0, effectiveness=0, cost=0, innovation=0,
            scalability=0, relevance=0, impact=0
        )
        max_total_rating = 0
        potential_ratings = []
        
        for i in range(5):
            print(f"Evaluating potential solution {i+1} of 5:")
            try:
                rating = evaluate_technology(row, target_problem, potential_solutions[i])
                
                # Verify we got a valid RatingResponse object
                if not isinstance(rating, RatingResponse):
                    print(f"Warning: Invalid rating type received. Using default rating.")
                    rating = RatingResponse(
                        feasibility=0, effectiveness=0, cost=0, innovation=0,
                        scalability=0, relevance=0, impact=0
                    )
                
                # Calculate total rating
                total_rating = sum([
                    rating.feasibility, rating.effectiveness, rating.cost,
                    rating.innovation, rating.scalability, rating.relevance,
                    rating.impact
                ]) // 7
                
                potential_ratings.append(rating)
                if total_rating > max_total_rating:
                    max_total_rating = total_rating
                    max_rating = rating
            except Exception as e:
                print(f"Error evaluating solution {i+1}: {e}")
                # Create a default rating for this solution
                default_rating = RatingResponse(
                    feasibility=0, effectiveness=0, cost=0, innovation=0,
                    scalability=0, relevance=0, impact=0
                )
                potential_ratings.append(default_rating)
        
        row['rating'] = max_rating
        row['potential_ratings'] = potential_ratings
    
    # After processing, sort technologies
    try:
        rated_techs = sorted(sampled_techs, key=lambda x: calculate_total_rating(x['rating']), reverse=True)
        
        # Filter technologies based on parameters selected earlier
        filtered_techs, report_title = filter_technologies(rated_techs, filter_params)
        
        if not filtered_techs:
            print("\nNo technologies met the filtering criteria.")
            return
            
        print(f"\n{report_title} from Randomly Sampled Set:")
        for idx, tech in enumerate(filtered_techs, start=1):
            tech_name = tech.get("Technology Name", "Unknown")
            total_rating = calculate_total_rating(tech['rating'])
            print(f"{idx}. {tech_name} (Rating: {total_rating})")
            
        # Create report with filtered technologies
        create_pdf_report(filtered_techs, target_problem, f"{report_title} (from Sample of {sample_size})")
        
    except Exception as e:
        print(f"Error sorting technologies: {e}")
        print("Attempting to continue with unsorted technologies...")
        top_count = min(5, len(sampled_techs))
        top_techs = sampled_techs[:top_count]
        create_pdf_report(top_techs, target_problem, "Technology Report (Unsorted)")

def get_filtering_parameters():
    """
    Ask the user how they want to filter the evaluated technologies for the final report.
    They can either specify the number of top technologies or set a minimum rating threshold.
    """
    print("\nHow would you like to filter technologies for the final report?")
    print("1. Select top N technologies by overall rating")
    print("2. Include only technologies above a minimum rating threshold")
    choice = input("Enter your choice (1/2): ").strip()
    
    if choice == "1":
        while True:
            try:
                num_techs = int(input("How many top technologies would you like to include? ").strip())
                if num_techs > 0:
                    return {"type": "top_n", "value": num_techs}
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
    elif choice == "2":
        while True:
            try:
                min_rating = int(input("Enter minimum rating threshold (1-100): ").strip())
                if 1 <= min_rating <= 100:
                    return {"type": "min_rating", "value": min_rating}
                else:
                    print("Please enter a number between 1 and 100.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        print("Invalid choice. Using default (top 5).")
        return {"type": "top_n", "value": 5}

def filter_technologies(rated_techs, filter_params):
    """
    Filter technologies based on user-defined parameters.
    """
    if filter_params["type"] == "top_n":
        # Return the top N technologies
        top_n = min(filter_params["value"], len(rated_techs))
        return rated_techs[:top_n], f"Top {top_n} Technologies"
    else:  # min_rating
        # Return technologies with ratings above the threshold
        min_rating = filter_params["value"]
        filtered_techs = [tech for tech in rated_techs if calculate_total_rating(tech['rating']) >= min_rating]
        return filtered_techs, f"Technologies with Rating >= {min_rating}"

def generate_report_title(target_problem, max_words=10):
    """
    Use the LLM to generate a concise title (max 10 words) for the report based on the target problem.
    """
    prompt = (
        f"Create a concise, descriptive title for a technology report addressing this problem: '{target_problem}'. "
        f"The title should be AT MOST {max_words} words and clearly represent the core issue. "
        "DO NOT use more than 10 words. Format as plain text without quotation marks."
    )
    
    try:
        # Remove the temperature parameter since it's not supported by ask_llm
        title = ask_llm(prompt, model=LLM_MODEL).strip()
        
        # Count the words to ensure we're within the limit
        word_count = len(title.split())
        if word_count > max_words:
            # If over the limit, truncate to the first max_words words
            title = " ".join(title.split()[:max_words])
            print(f"Title was truncated to {max_words} words: '{title}'")
        
        # Remove any quotation marks that might have been added
        title = title.replace('"', '').replace("'", "")
        
        print(f"Generated report title: '{title}'")
        return title
    except Exception as e:
        print(f"Error generating title: {e}")
        # Fallback to a default title
        return f"Tech Report on {' '.join(target_problem.split()[:5])}"

def create_markdown_report(top_techs, target_problem, report_title="Top Technologies Analysis", report_folder="reports"):
    """
    Create a markdown report, saved in the reports folder.
    Uses an LLM-generated concise title for the filename and main heading.
    Includes a dedicated problem statement section.
    """
    from pathlib import Path
    
    # Create reports directory if it doesn't exist
    Path(report_folder).mkdir(parents=True, exist_ok=True)
    
    # Generate a concise title for the file
    concise_title = generate_report_title(target_problem)
    
    # Create a sanitized filename from the concise title
    safe_filename = "".join(c if c.isalnum() else "_" for c in concise_title)
    md_filename = Path(report_folder) / f"{safe_filename}.md"
    
    with open(md_filename, "w", encoding="utf-8") as md_file:
        # Use the concise title as the main document title
        md_file.write(f"# {concise_title}\n\n")
        
        # Add a clear problem statement section
        md_file.write("## Problem Statement\n\n")
        md_file.write(f"{target_problem}\n\n")
        
        # Add report type as a subheading
        md_file.write(f"## {report_title}\n\n")
        
        # Create an individual section for each technology
        for idx, tech in enumerate(top_techs, start=1):
            print(f"Processing technology {idx} of {len(top_techs)}")
            
            tech_name = tech.get('Technology Name', 'N/A')
            solution = tech.get('Solution', 'No solution provided')
            
            # Section heading
            md_file.write(f"### Technology {idx}: {tech_name}\n\n")
            
            # Technology solution
            md_file.write(f"**Solution:** {solution}\n\n")
            
            # NEW: Simple explanation of the technology itself
            md_file.write("#### Technology Overview (High School Level):\n\n")
            tech_explanation = getSimpleTechExplanation(tech_name, solution, target_problem)
            md_file.write(f"{tech_explanation}\n\n")
            
            # Rating section
            rating = tech.get('rating')
            if rating:
                md_file.write("#### Rating Breakdown:\n\n")
                md_file.write(f"- **Feasibility**: {rating.feasibility}/100\n")
                md_file.write(f"- **Effectiveness**: {rating.effectiveness}/100\n")
                md_file.write(f"- **Cost Efficiency**: {rating.cost}/100\n")
                md_file.write(f"- **Innovation**: {rating.innovation}/100\n")
                md_file.write(f"- **Scalability**: {rating.scalability}/100\n")
                md_file.write(f"- **Relevance**: {rating.relevance}/100\n")
                md_file.write(f"- **Impact**: {rating.impact}/100\n")
                md_file.write(f"- **Overall Score**: {calculate_total_rating(rating)}/100\n\n")
            
            # Display for each potential solution an analysis generated by the LLM
            for i in range(5):
                print(f"Processing potential solution {i+1} of 5")
                potential_solution = tech.get('potential_solutions', [])[i]
                rating = tech.get('potential_ratings', [])[i]
                
                if potential_solution:
                    # Potential solution header
                    md_file.write(f"#### Potential Solution {i+1}:\n\n")
                    
                    # The potential solution itself
                    md_file.write(f"*{potential_solution}*\n\n")
                    
                    # High School Level Explanation
                    md_file.write("##### Simple Explanation (High School Level):\n\n")
                    simple_explanation = getSimplifiedExplanation(potential_solution, target_problem)
                    md_file.write(f"{simple_explanation}\n\n")
                    
                    # NEW: Pros and Cons for the solution
                    md_file.write("##### Pros and Cons:\n\n")
                    pros_cons = getSolutionProsAndCons(potential_solution, target_problem)
                    md_file.write(f"{pros_cons}\n\n")
                    
                    # Solution Effectiveness for This Problem
                    md_file.write("##### Why This Works For The Problem:\n\n")
                    solution_effectiveness = getSolutionEffectiveness(potential_solution, target_problem)
                    md_file.write(f"{solution_effectiveness}\n\n")
                    
                    # Analysis header (focus on general strengths/weaknesses)
                    md_file.write("##### Detailed Analysis:\n\n")
                    analysis = getSolutionAnalysis(potential_solution, rating, target_problem)
                    md_file.write(f"{analysis}\n\n")
                    
                    # Next steps header
                    md_file.write("##### Implementation Steps:\n\n")
                    next_steps = getNextSteps(potential_solution, target_problem)
                    md_file.write(f"{next_steps}\n\n")
                    
                    # Add a horizontal rule after each solution except the last one
                    if i < 4:
                        md_file.write("---\n\n")
            
            # Add a horizontal rule after each technology except the last one
            if idx < len(top_techs):
                md_file.write("\n\n---\n\n")
    
    print(f"Markdown report created: {md_filename}")
    return str(md_filename)

def convert_markdown_to_pdf(markdown_file, report_folder="reports"):
    """
    Convert the markdown file to PDF using the markdown-pdf library.
    Store all reports in a dedicated reports folder.
    
    Args:
        markdown_file: Path to the markdown file
        report_folder: Folder to store reports (default: "reports")
    
    Returns:
        Path to the generated PDF file or None if conversion failed
    """
    try:
        from markdown_pdf import MarkdownPdf, Section
        from pathlib import Path
        
        # Create reports directory if it doesn't exist
        Path(report_folder).mkdir(parents=True, exist_ok=True)
        
        # Get the base filename without extension
        base_name = Path(markdown_file).stem
        
        # Define output PDF path in the reports folder
        pdf_filename = Path(report_folder) / f"{base_name}.pdf"
        
        print(f"Converting markdown to PDF: {pdf_filename}")
        
        # Read the markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Create a PDF with table of contents up to level 3
        pdf = MarkdownPdf(toc_level=3)
        
        # Add the markdown content as a section
        pdf.add_section(Section(markdown_content))
        
        # Set document metadata
        pdf.meta["title"] = base_name
        pdf.meta["author"] = "Technology Rating Evaluator"
        
        # Save the PDF to the specified path
        pdf.save(str(pdf_filename))
        
        print(f"PDF report created: {pdf_filename}")
        return str(pdf_filename)
        
    except ImportError as e:
        print(f"Missing library: {e}")
        print("Please install required library: pip install markdown-pdf")
        return None
    except Exception as e:
        print(f"Error converting markdown to PDF: {e}")
        print(f"Exception details: {type(e).__name__}: {str(e)}")
        return None

def getSimpleTechExplanation(tech_name, solution, target_problem):
    """
    Generate a high-school level explanation of the technology itself.
    """
    prompt = (
        f"Explain the technology '{tech_name}' and its solution '{solution}' in simple terms that a high school student "
        f"would understand. "
        "Avoid technical jargon, use simple analogies where helpful, and explain the basic concept in about 2-3 paragraphs. "
        "Format your response in markdown with a conversational tone."
    )
    return ask_llm(prompt, model=LLM_MODEL)

def getSolutionProsAndCons(potential_solution, target_problem):
    """
    Generate a simple bullet list of pros and cons for the solution.
    """
    prompt = (
        f"For this potential solution: '{potential_solution}' addressing the problem: '{target_problem}', "
        "create a clear list of pros and cons. Identify 3-5 specific advantages and 3-5 potential drawbacks or limitations. "
        "Format your response as markdown bullet lists with the headings '**Pros:**' and '**Cons:**'."
    )
    return ask_llm(prompt, model=LLM_MODEL)

def main():
    print("=== Welcome to the Technology Rating Evaluator ===")
    csv_file_path = input("Enter CSV file path (default 'VentureLabData.csv'): ").strip()
    if not csv_file_path:
        csv_file_path = "VentureLabData.csv"

    techs = load_technologies(csv_file_path)
    if not techs:
        print("No technology records found. Please check the CSV file path and format.")
        return

    while True:
        print("\nInteractive Menu:")
        print("1. Generate a report for all technologies")
        print("2. Generate a report for a random sample of technologies")
        print("3. Exit")
        choice = input("Enter your option (1/2/3): ").strip()
        
        if choice == "1":
            automatic_all(techs)
        elif choice == "2":
            random_sample_technologies(techs)
        elif choice == "3":
            print("Exiting the evaluator.")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
