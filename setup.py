from setuptools import setup, find_packages

setup(
    name="tms_routing",           # 패키지 이름
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},      # 소스코드가 src/ 안에 있으므로 지정
    install_requires=[
        "pymysql",
        "sqlalchemy",
        "pandas",
        "geopy",
        "scikit-learn",
        "numpy",
    ],
)