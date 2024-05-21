from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np



def plot_learning_curve(scores, file):
    x = [i+1 for i in range(len(scores))]
    avg_scores = np.zeros(len(scores))
    
    for i in range(len(avg_scores)):
        avg_scores[i] = np.mean(scores[max(0, i-100):(i+1)])
        
    plt.plot(x, avg_scores)
    plt.title("average score of previous 100 steps")
    
    plt.savefig(file)

def save_frames_as_gif(frames, file):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(file, writer='imagemagick', fps=60)
