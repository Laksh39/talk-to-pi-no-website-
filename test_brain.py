
import ollama

# We'll use the Llama model for this one
response = ollama.chat(model='llama3.2:1b', messages=[
  {
    'role': 'user',
    'content': 'Explain why the sky is blue like I am 5 years old.',
  },
])

print("-" * 30)
print("LOCAL BRAIN SAYS:")
print(response['message']['content'])
print("-" * 30)

