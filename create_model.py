import language_model
print("Generating neural networks...")
networks = language_model.generate_networks()
language_model.conversion_net = networks["conversion"]
language_model.generation_net = networks["generation"]
print("Neural networks generated.")
print("Saving LM...")
language_model.save_models()
print("LM saved.")