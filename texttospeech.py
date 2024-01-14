from transformers import pipeline

def read_paragraph(paragraph):
    # Load the text generation pipeline
    text_generator = pipeline("text-generation")

    # Generate text based on the input paragraph
    generated_text = text_generator(paragraph, max_length=150, num_return_sequences=1)[0]['generated_text']

    return generated_text

# Example usage
paragraph = "Hugging Face is an AI research organization that focuses on natural language processing."
result = read_paragraph(paragraph)
print(result)
