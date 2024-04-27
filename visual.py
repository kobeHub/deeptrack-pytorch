from matplotlib import pyplot as plt


def visual_one_sample(img):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img[0], cmap='gray')
    axs[0].set_title('The Observable Object Edges')
    axs[0].plot(25, 50, 'ro')  # Plot robot as a red dot at (25, 0)
    axs[0].set_xlim(0, 50)
    axs[0].set_ylim(0, 50)

    axs[1].imshow(img[1], cmap='gray')
    axs[1].set_title('The Observable Scene from Robot')
    axs[1].plot(25, 50, 'ro')  # Plot robot as a red dot at (25, 0)
    axs[1].set_xlim(0, 50)
    axs[1].set_ylim(0, 50)

    plt.show()


def visual_multiple_samples(imgs):
    """
    Args:
     imgs: a tensor of shape (batch_size, 2, 50, 50)
    """
    # Create a 5x3 grid of subplots
    # Adjusted to 5x6 for two images per sample
    fig, axs = plt.subplots(2, 5, figsize=(18, 10))
    for i, img in enumerate(imgs):
        row = i // 5
        col = i % 5  # Multiply by 2 to leave space for the second channel
        print(f'Row: {row}, Col: {col}')

        # Display the first channel of the image
        axs[row, col].imshow(img[1], cmap='gray')
        axs[row, col].set_title(f'The Observable Scene at step {i}')
        axs[row, col].plot(25, 50, 'ro')  # Plot robot as a red dot at (25, 50)
        axs[row, col].set_xlim(0, 50)
        axs[row, col].set_ylim(0, 50)

        # Display the second channel of the image
        # axs[row, col+1].imshow(img[1], cmap='gray')
        # axs[row, col+1].set_title('The Observable Scene from Robot')
        # # Plot robot as a red dot at (25, 50)
        # axs[row, col+1].plot(25, 50, 'ro')
        # axs[row, col+1].set_xlim(0, 50)
        # axs[row, col+1].set_ylim(0, 50)

    plt.tight_layout()
    plt.show()
    fig.savefig('results/imgs/input_seq.png', dpi=300)


def plot_losses(training_loss, img_path):
    # Visualize the training loss
    plt.figure()
    plt.plot(training_loss)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.savefig(img_path, dpi=300)
