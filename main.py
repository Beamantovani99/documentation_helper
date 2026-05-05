from dotenv import load_dotenv
import os
def main():
    print("Hello from langchain-course-main!")
    print(os.environ.get("OPENAI_API_KEY"))

if __name__ == "__main__":
    main()
