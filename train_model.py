import language_model
print("Loading neural networks...")
networks = language_model.load_networks()
print("Neural networks loaded.")
language_model.conversion_net = networks["conversion"]
language_model.generation_net = networks["generation"]
print("Starting training...")
language_model.train(language_model.training_set, language_model.generation_net, 1)
print("Training complete.")
print("Saving models...")
language_model.save_models()
print("Models saved. Program complete.")