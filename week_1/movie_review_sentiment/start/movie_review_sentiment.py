from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

def analyze_sentiment(review):
    """
    Analyze the sentiment of a movie review using structured output.
    Returns a dictionary with 'thought' and 'sentiment' keys.
    """
    # TODO: Create a prompt that:
    # 1. Asks for sentiment analysis
    # 2. Specifies the required output format
    #       thought: [analysis]
    #       sentiment: [positive/negative]
    # 3. Includes the review text
    prompt = f"""
    You are a sentiment analysis expert. Analyze the sentiment of the following movie review.
    
    Instructions:
    - Provide a brief analysis of the reviewer's opinion and reasoning
    - Determine if the overall sentiment is positive or negative
    - Respond ONLY in the exact format below with no additional text, explanation, or commentary:
    
    thought: [your analysis here]
    sentiment: [positive or negative]
    
    Review to analyze:
    {review}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    content = response.choices[0].message.content
    
    # Parse the response to extract thought and sentiment
    # The response should be in the format:
    # thought: [analysis]
    # sentiment: [positive/negative]
    result = {
        "thought": "",
        "sentiment": ""
    }
    
    # Split by lines and parse
    for line in content.strip().split('\n'):
        line = line.strip()
        if line.startswith('thought:'):
            result["thought"] = line.replace('thought:', '', 1).strip()
        elif line.startswith('sentiment:'):
            result["sentiment"] = line.replace('sentiment:', '', 1).strip()
    
    return result

def main():
    # Test cases
    reviews = [
        "This film shouldn't work at all. It doesn't have much of a story and the whole dial up internet thing is incredibly dated. However Hanks and Ryan sell it beautifully.",
        "The movie was terrible. The acting was wooden, the plot made no sense, and I want my two hours back.",
        "An absolute masterpiece! The cinematography was stunning, the acting was superb, and the story kept me engaged from start to finish."
    ]

    # Test each review
    for i, review in enumerate(reviews, 1):
        result = analyze_sentiment(review)
        print(f"\nReview {i}:")
        print(f"Thought: {result['thought']}")
        print(f"Sentiment: {result['sentiment']}")

if __name__ == "__main__":
    main() 