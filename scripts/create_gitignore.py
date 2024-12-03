def create_gitignore():
    """Create .gitignore file with appropriate rules."""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Data
data/reich_lab/
data/raw/
*.fasta
*.fastq
*.bam
*.sam
*.vcf

# Logs and checkpoints
logs/
checkpoints/
*.log
*.pt
*.pth

# IDE
.idea/
.vscode/
*.swp
.DS_Store

# Environment
.env
.venv
env/
venv/
ENV/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("Created .gitignore file")

if __name__ == "__main__":
    create_gitignore() 