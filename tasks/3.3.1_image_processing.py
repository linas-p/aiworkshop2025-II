import ollama

response = ollama.chat(
    model='gemma3',
    messages=[{
        'role': 'user',
        'content': 'Kas matoma paveikslėlyje?',
        'images': ['image.jpg']
    }]
)

print(response)