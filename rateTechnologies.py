import csv
from pydantic import BaseModel
from askLlama import ask_llm_with_schema

# Define a Pydantic model for collecting the rating from the LLM.
class RatingResponse(BaseModel):
    rating: int

def evaluate_technology(row, target_problem):
    """
    Construct a prompt from the technology details and the user-supplied target problem.
    Call ask_llm_with_schema to get a rating (1-100) from the LLM based on how well
    the provided solution addresses the target problem.
    """
    tech_name = row.get("Technology Name", "Unknown Technology")
    solution = row.get("Solution", "No solution provided")
    
    # Prepare the prompt with the user's target problem.
    question = (
        f"Technology '{tech_name}' proposes the following solution: '{solution}'. "
        f"Consider the problem: '{target_problem}'. "
        "On a scale of 1-100, where 1 means the solution is completely irrelevant and "
        "100 means it is perfectly tailored to solve the problem, please provide a rating of "
        "how effective or pertinent the solution is in addressing the problem. "
        "Return only a JSON object in the following format: {\"rating\": <your rating>}."
    )
    
    rating_response = ask_llm_with_schema(question, RatingResponse, model="llama3.2", temperature=0)
    
    try:
        rating_value = rating_response.rating
    except Exception as e:
        print(f"Error processing rating for technology '{tech_name}': {e}")
        rating_value = 0
    return rating_value

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
    The user is first prompted for a target problem, and then each technology's solution
    is evaluated against that target problem without further individual prompts.
    After processing, the top 5 highest-rated technologies are displayed.
    """
    target_problem = input("Enter the target problem you want solutions to address: ").strip()
    for index, row in enumerate(techs, start=1):
        print(f"\nEvaluating Technology {index}:")
        rating = evaluate_technology(row, target_problem)
        row['rating'] = rating
        print(f"Rating for '{row.get('Technology Name', 'N/A')}': {rating}")

    # After processing, sort and display the top 5 technologies.
    rated_techs = sorted(techs, key=lambda x: int(x.get('rating', 0)), reverse=True)
    print("\nTop 5 Most Helpful Technologies Based on LLM Rating:")
    top5 = rated_techs[:5]
    for rank, tech in enumerate(top5, start=1):
        print(f"\nRank {rank}:")
        print(f"Technology Name: {tech.get('Technology Name', 'N/A')}")
        print(f"Solution: {tech.get('Solution', 'N/A')}")
        print(f"Rating: {tech.get('rating', 0)}")

def interactive_search(techs):
    """
    Allow the user to search for a technology by name and then evaluate the technology's solution
    based on a target problem provided interactively.
    """
    search_str = input("Enter Technology Name to search (partial names allowed): ").strip().lower()
    matches = [row for row in techs if search_str in row.get("Technology Name", "").lower()]

    if not matches:
        print("No matching technology found.")
        return

    print(f"\nFound {len(matches)} matching technology(ies):")
    for idx, row in enumerate(matches, start=1):
        print(f"{idx}. {row.get('Technology Name', 'N/A')} - Solution: {row.get('Solution', 'N/A')}")
    
    selection = input("Enter the number of the technology to evaluate: ").strip()
    try:
        selected_idx = int(selection) - 1
        if 0 <= selected_idx < len(matches):
            selected_tech = matches[selected_idx]
            target_problem = input("Enter the target problem you want this technology's solution to address: ").strip()
            print(f"\nEvaluating Technology: {selected_tech.get('Technology Name', 'N/A')}")
            rating = evaluate_technology(selected_tech, target_problem)
            selected_tech['rating'] = rating
            print(f"Rating: {rating}")
        else:
            print("Invalid selection.")
    except Exception:
        print("Invalid input.")

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
        print("1. Automatically evaluate all technologies")
        print("2. Evaluate a specific technology by name")
        print("3. Exit")
        choice = input("Enter your option (1/2/3): ").strip()
        if choice == "1":
            automatic_all(techs)
        elif choice == "2":
            interactive_search(techs)
        elif choice == "3":
            print("Exiting the evaluator.")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
