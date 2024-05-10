import numpy as np

def Acceptance_ratio(Unserved, events):
    Tot_Users = Unserved.shape[0]
    # AcceptanceRatio = np.zeros((1, Tot_Users))S
    Tot_requests = np.zeros((Tot_Users))
    
    for l in range(events.shape[0]):
        if events[l, 0] < np.max(events[:, 0]) and events[l, 0] > 0:
            Tot_requests[events[l, 1]-1] += 1
    
    AcceptanceRatio = Tot_requests - Unserved
    Tot_requests[Tot_requests == 0] = 1
    AcceptanceRatio = (AcceptanceRatio * 100. / Tot_requests)
    
    # Code for plotting, uncomment if needed
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 6))
    # plt.bar(np.arange(Tot_Users), AcceptanceRatio[0], color='blue')
    # plt.xlabel('User ID')
    # plt.ylabel('Request Acceptance Ratio')
    # plt.grid(True)
    # plt.show()

    return AcceptanceRatio
if __name__ =="__main__":
# Example usage:
    Unserved = np.array([[0, 1, 0], [1, 0, 1]])  # Example array for Unserved
    events = np.array([[1, 0], [2, 1], [3, 0], [4, 2], [5, 0]])  # Example array for events
    AcceptanceRatio = Acceptance_ratio(Unserved, events)
    print(AcceptanceRatio)
