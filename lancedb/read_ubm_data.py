import lmdb
import sys

def read_lmdb(db_path):
    # Open the LMDB environment in read-only mode
    env = lmdb.open(db_path, readonly=True, lock=False)

    # Start a read-only transaction
    with env.begin() as txn:
        # Create a cursor to iterate over the database
        cursor = txn.cursor()
        
        # Iterate through all key-value pairs
        print("Reading all key-value pairs:")
        for key, value in cursor:
            # Decode bytes to strings if needed (assuming keys/values are strings)
            try:
                key_str = key.decode('utf-8')
                value_str = value.decode('utf-8')
                print(f"Key: {key_str}, Value: {value_str}")
            except UnicodeDecodeError:
                # Handle cases where data is not UTF-8 encoded
                print(f"Key: {key}, Value: {value}")

        # Example: Get a specific key
        specific_key = b'key1'  # Replace with the key you want to look up
        value = txn.get(specific_key)
        if value:
            try:
                value_str = value.decode('utf-8')
                print(f"\nValue for {specific_key.decode('utf-8')}: {value_str}")
            except UnicodeDecodeError:
                print(f"\nValue for {specific_key}: {value}")
        else:
            print(f"\nKey {specific_key} not found.")

    # Environment is automatically closed when exiting the 'with' block

if __name__ == "__main__":
    read_lmdb("ubm_data")