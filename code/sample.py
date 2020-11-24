import numpy as np
import torch

def main():
    a = np.arange(12).reshape(1,12)
    print(a)
    
    b = torch.tensor(a)
    print(b)
    
    c = torch.squeeze(b, 0)
    print(c)

if __name__ == "__main__":
    main()