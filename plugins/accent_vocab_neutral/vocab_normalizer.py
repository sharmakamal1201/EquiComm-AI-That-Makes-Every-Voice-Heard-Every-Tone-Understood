import os
from openai import AzureOpenAI
 
key = ""
endpoint = os.getenv("ENDPOINT_URL", "https://equicomai.cognitiveservices.azure.com/")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4.1")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", key)
 
# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

def normalize_vocabulary(input_text):
    """
    Normalize regional idioms and colloquialisms to globally understood English
    
    Args:
        input_text (str): The text to normalize
        
    Returns:
        dict: Contains 'original', 'alternate', 'changed' keys
    """
    
    # Prepare the chat prompt with the vocab normalization task
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """Task:
 
Rewrite the input text only if it contains region-specific idioms, colloquialisms, or phrases (Indian, British, American) that may cause misunderstanding across regions.
 
 
Rules:
 
> Scope: Replace phrases only if they differ across Indian, British, or American English.
 
> Workplace idioms: Replace only if regional interpretations differ (e.g., "table the discussion").
 
> Ignore: Universally understood idioms.
 
> Replacements: Normalize to the least ambiguous global equivalent.
 
> Evaluation: Each occurrence is evaluated independently (same phrase may map differently in different contexts).
 
> Preserve: Capitalization and tense.
 
> Time expressions: Convert "half five" → "5:30".
 
 
Ambiguity:
 
> If context resolves → replace.
 
> If ambiguous everywhere → leave unchanged.
 
 
Output format:
(always exactly 3 lines, no commentary)
 
Original: <input text>
Alternate: <rewritten text if changes, else repeat Original>
Changed: <CSV list of replacements OR 'None'>
 
 
Block treatment:
 
> Multi-sentence input is one block.
 
> Always return one Original, one Alternate, one Changed.
 
 
Examples:
 
Example 1 – compound replacements
 
Original: He passed out after the party. He also passed out of college last year.
Alternate: He fainted after the party. He also graduated last year.
Changed: passed out → fainted, passed out of college → graduated
 
 
Example 2 – workplace idiom difference
 
Original: The manager asked us to table the discussion. We'll revisit it tomorrow.
Alternate: The manager asked us to postpone the discussion. We'll revisit it tomorrow.
Changed: table the discussion → postpone the discussion
 
 
Example 3 – time expression
 
Original: I'll call you at half five. Let's review the document before then.
Alternate: I'll call you at 5:30. Let's review the document before then.
Changed: half five → 5:30
 
 
Example 4 – no changes (better "None" case, multi-sentence)
 
Original: We discussed the project yesterday. Let's review again next week.
Alternate: We discussed the project yesterday. Let's review again next week.
Changed: None"""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": input_text
                }
            ]
        }
    ]
    
    try:
        # Generate the completion
        completion = client.chat.completions.create(
            model=deployment,
            messages=chat_prompt,
            max_tokens=300,
            temperature=0.1,  # Low temperature for consistent formatting
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        
        # Parse the response
        response_text = completion.choices[0].message.content.strip()
        lines = response_text.split('\n')
        
        # Extract the three required lines
        result = {
            'original': '',
            'alternate': '',
            'changed': ''
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Original:'):
                result['original'] = line.replace('Original:', '').strip()
            elif line.startswith('Alternate:'):
                result['alternate'] = line.replace('Alternate:', '').strip()
            elif line.startswith('Changed:'):
                result['changed'] = line.replace('Changed:', '').strip()
        
        return result
        
    except Exception as e:
        # Return input unchanged if there's an error
        return {
            'original': input_text,
            'alternate': input_text,
            'changed': 'Error: ' + str(e)
        }


# Test function
if __name__ == "__main__":
    # Test with a sample phrase
    test_text = "Let's table this discussion and circle back later."
    result = normalize_vocabulary(test_text)
    
    print("=== Vocab Normalization Test ===")
    print(f"Original: {result['original']}")
    print(f"Alternate: {result['alternate']}")
    print(f"Changed: {result['changed']}")