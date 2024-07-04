import matplotlib.pyplot as plt
from pdf2image import convert_from_path

# Paths to your images
path1 = "/home/saandeepaath-admin/projects/ABAW/flowcon_ood_heatmap/data/plots/main/cifar10_3.pdf"
path2 = "/home/saandeepaath-admin/projects/ABAW/flowcon_ood_heatmap/data/plots/main/cifar10_3_cifar100.pdf"
path3 = "/home/saandeepaath-admin/projects/ABAW/flowcon_ood_heatmap/data/plots/main/cifar10_7.pdf"
path4 = "/home/saandeepaath-admin/projects/ABAW/flowcon_ood_heatmap/data/plots/main/cifar10_7_cifar100.pdf"

path11 = "/home/saandeepaath-admin/projects/ABAW/flowcon_ood_heatmap/data/plots/main/cifar100_3.pdf"
path12 = "/home/saandeepaath-admin/projects/ABAW/flowcon_ood_heatmap/data/plots/main/cifar100_3_cifar10.pdf"
path13 = "/home/saandeepaath-admin/projects/ABAW/flowcon_ood_heatmap/data/plots/main/cifar100_6.pdf"
path14 = "/home/saandeepaath-admin/projects/ABAW/flowcon_ood_heatmap/data/plots/main/cifar100_6_cifar10.pdf"

pdf_paths = [path1, path2, path3, path4]
# pdf_paths = [path1, path2, path3, path4]
dpi_value = 300  
# Convert the first page of each PDF to an image
images = [convert_from_path(pdf_path, first_page=0, last_page=1,  dpi=dpi_value)[0] for pdf_path in pdf_paths]

# Create a 2x2 grid for the images
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

title = ['(a)', '(b)', '(c)', '(d)']
# Display each image in its respective subplot
for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i], aspect='auto')
    ax.axis('off')  # Hide axis
    ax.set_xlabel(title[i], fontsize=36)

# Adjust layout
plt.tight_layout()

# Save the plot as a PDF
plt.savefig('pdfs_grid.pdf')

# Optionally, display the plot
plt.show()
