import tiktoken

def decode_tokens():
    enc = tiktoken.get_encoding("gpt2")
    
    print("--- Atomic-1Bit Decoder ---")
    print("Paste the token IDs from the C++ engine below (space separated):")
    
    try:
        raw = input("> ")
        tokens = [int(t) for t in raw.split()]
        text = enc.decode(tokens)
        print("\n[Decoded Text]:")
        print(text)
        print("------------------")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    decode_tokens()
