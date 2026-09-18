from glob import glob
from setuptools import setup


package_name = "narrow_memory_nav"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name, "narrow_memory_core"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xiaotian",
    maintainer_email="zouxiaotian12@gmail.com",
    description="Failure-aware narrow passage memory wrapper for ROS 2 navigation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "memory_decision_node = narrow_memory_nav.memory_decision_node:main",
            "safety_fusion_node = narrow_memory_nav.safety_fusion_node:main",
            "fsm_nav_node = narrow_memory_nav.fsm_nav_node:main",
        ],
    },
)
