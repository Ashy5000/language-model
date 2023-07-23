import language_model
networks = language_model.load_networks()
language_model.conversion_net = networks["conversion"]
language_model.generation_net = networks["generation"]
language_model.train(language_model.training_set, language_model.generation_net, 1)
language_model.save_models()