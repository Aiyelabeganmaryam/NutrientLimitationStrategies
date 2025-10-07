# Commands to run AFTER creating GitHub repository

# 1. First, create the repository manually on GitHub.com with these settings:
#    - Name: NutrientLimitationStrategies
#    - Public repository
#    - Don't initialize with README, .gitignore, or license

# 2. Then replace YOURUSERNAME below with your actual GitHub username:

git remote add origin https://github.com/YOURUSERNAME/NutrientLimitationStrategies.git

# 3. Push your code to GitHub:
git push -u origin main

# 4. Verify it worked:
git remote -v

# Example: If your GitHub username is "john_doe", the command would be:
# git remote add origin https://github.com/john_doe/NutrientLimitationStrategies.git

# After successful push, your repository will be at:
# https://github.com/YOURUSERNAME/NutrientLimitationStrategies