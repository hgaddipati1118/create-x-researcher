import csv
import random  # Add this import for random sampling
from pydantic import BaseModel
from askLlama import ask_llm, ask_llm_with_schema

# Define the LLM model to use throughout the application
LLM_MODEL = "qwen2.5:3b"

# New imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

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

# Add this helper function for calculating total rating safely
def calculate_total_rating(rating):
    """Calculate total rating from a RatingResponse object safely."""
    try:
        if isinstance(rating, RatingResponse):
            return sum([
                rating.feasibility, rating.effectiveness, rating.cost,
                rating.innovation, rating.scalability, rating.relevance,
                rating.impact
            ]) // 7
        else:
            print(f"Warning: Invalid rating type: {type(rating)}. Using 0.")
            return 0
    except Exception as e:
        print(f"Error calculating rating: {e}")
        return 0

def create_pdf_report(top_techs, target_problem, report_title="Top Technologies Analysis"):
    """
    Create a McKinsey-style PDF report with an in-depth section for each technology.
    Each section displays the technology's solution, rating, and includes an analysis generated by the LLM
    that brainstorms ways the solution could help address the target problem.
    """
    doc = SimpleDocTemplate(f"ReportOfTechnologiesFor{target_problem}.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Report Title
    title = Paragraph(f"{report_title} for {target_problem}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 24))

    # Create an individual section for each technology
    for idx, tech in enumerate(top_techs, start=1):
        print(f"Processing technology {idx} of {len(top_techs)}")
        # Section heading with rank and technology name
        heading_text = f"Section {idx}: {tech.get('Technology Name', 'N/A')}"
        heading = Paragraph(heading_text, styles['Heading2'])
        story.append(heading)
        story.append(Spacer(1, 12))

        # Display the solution details
        solution_details = f"<b>Solution:</b> {tech.get('Solution', 'No solution provided')}"
        solution_paragraph = Paragraph(solution_details, styles['BodyText'])
        story.append(solution_paragraph)
        story.append(Spacer(1, 12))

        # Display the rating details
        rating_details = f"<b>Rating:</b> {tech.get('rating', 0)}"
        rating_paragraph = Paragraph(rating_details, styles['BodyText'])
        story.append(rating_paragraph)
        story.append(Spacer(1, 12))

        # Display for each potential solution an analysis generated by the LLM
        for i in range(5):
            print(f"Processing potential solution {i+1} of 5")
            potential_solution = tech.get('potential_solutions', [])[i]
            rating = tech.get('potential_ratings', [])[i]
            analysis_details = getSolutionAnalysis(potential_solution, rating, target_problem)
            potential_solution_details = f"<b>Potential Solution {i+1}:</b> {potential_solution}"
            potential_solution_paragraph = Paragraph(potential_solution_details, styles['BodyText'])
            story.append(potential_solution_paragraph)
            story.append(Spacer(1, 12))
            analysis_paragraph = Paragraph(analysis_details, styles['BodyText'])
            story.append(analysis_paragraph)
            story.append(Spacer(1, 12))
            next_steps = getNextSteps(potential_solution, target_problem)
            next_steps_paragraph = Paragraph(next_steps, styles['BodyText'])
            story.append(next_steps_paragraph)
            story.append(Spacer(1, 12))

    # Add a conclusion section
    conclusion = Paragraph(
        "The above report provides a detailed analysis of the top 5 technologies, highlighting the strengths and "
        "strategic advantages of each solution. The analyses are based on advanced language model insights that "
        "brainstorm several distinct ways each technology can address the target problem.",
        styles['BodyText']
    )
    story.append(conclusion)

    # Build the PDF report
    doc.build(story)
    print(f"{report_title} Report '{target_problem}.pdf' has been created.")

def getSolutionAnalysis(potential_solution, rating, target_problem):
    """
    Generate an analysis using LLM that explains how the potential solution could help solve the target problem.
    """
    prompt = (
        f"Consider the following potential solution: '{potential_solution}'. "
        f"Given the target problem: '{target_problem}', please explain how the rating of the solution is: {rating}. "
        f"The rating is on a scale of 1-100, where 1 is the worst and 100 is the best. And also the rating is based on the following criteria: feasibility, effectiveness, cost, innovation, scalability, relevance, impact. "
        "Provide a detailed analysis of how this solution addresses the problem. Have it be under 2000 tokens."
    )
    return ask_llm(prompt, model=LLM_MODEL)

def getNextSteps(potential_solution, target_problem):
    """
    Generate next steps using LLM that suggests how to implement the potential solution.
    """
    prompt = (
        f"Consider the following potential solution: '{potential_solution}'. "
        f"Given the target problem: '{target_problem}', please suggest how to implement the solution and give the next steps to take considering that someone is just starting on this solution. Have it be under 2000 tokens and be concise"
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
