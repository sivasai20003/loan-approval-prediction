# Create the ZIP file containing all project files
import zipfile
import os

def create_project_zip():
    """Create a ZIP file with all project components"""
    
    # List of files to include in the ZIP
    files_to_zip = [
        'loan_approval_prediction.py',
        'Loan_Approval_Prediction.ipynb', 
        'README.md',
        'requirements.txt',
        'presentation_guide.md',
        'quick_demo.py'
    ]
    
    # Create ZIP file
    zip_filename = 'loan_approval_prediction_project.zip'
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                zipf.write(file)
                print(f"✅ Added {file} to ZIP")
            else:
                print(f"❌ Warning: {file} not found")
    
    # Check ZIP contents
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        zip_contents = zipf.namelist()
    
    print(f"\n📦 ZIP FILE CREATED: {zip_filename}")
    print(f"📁 Contains {len(zip_contents)} files:")
    for i, file in enumerate(zip_contents, 1):
        print(f"   {i}. {file}")
    
    # Get file size
    zip_size = os.path.getsize(zip_filename)
    print(f"\n💾 File size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
    
    return zip_filename, zip_contents

# Create the ZIP file
zip_name, contents = create_project_zip()

print(f"\n🎉 PROJECT PACKAGE READY!")
print("=" * 40)
print("📦 Your complete Loan Approval Prediction project")
print("   is packaged in: loan_approval_prediction_project.zip")
print()
print("📋 What's included:")
print("   • Complete Python script (ready to run)")
print("   • Jupyter notebook version")
print("   • Professional README documentation")
print("   • Requirements file for dependencies")
print("   • Presentation guide for interviews")
print("   • Quick demo script")
print()
print("🚀 Next Steps:")
print("   1. Download the ZIP file")
print("   2. Extract and run the code")
print("   3. Add to your GitHub repository")
print("   4. Include in your portfolio")
print("   5. Add to resume under 'Projects' section")
print("=" * 40)