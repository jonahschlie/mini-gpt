def load_data():
    test_string = "low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new"
    full_data = open('data/Shakespeare_clean_full.txt', 'r').read()
    training_data = open('data/Shakespeare_clean_train.txt', 'r').read()
    test_data = open('data/Shakespeare_clean_test.txt', 'r').read()
    valid_data = open('data/Shakespeare_clean_valid.txt', 'r').read()

    return test_string, full_data, training_data, test_data, valid_data