from loader import DataLoader
# from parafac import 


def main():
    file_path = "u.data"
    loader = DataLoader(file_path, nrows=100)
    loader.display_shape()
    print(f"Train data shape: {loader.train_data.shape}")
    print(f"Test data shape: {loader.test_data.shape}")


if __name__ == "__main__":
    main()