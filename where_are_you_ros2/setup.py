from setuptools import setup

package_name = 'where_are_you_ros2'

setup(
    name=package_name,
    version='0.20.5',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Luca Beber',
    author_email='luca.beber@gmail.com',
    maintainer='Luca Beber',
    maintainer_email='luca.beber@gmail.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Python action tutorials code.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'where_are_you_action_server = where_are_you_ros2.where_are_you_action_server:main',
            'where_are_you_action_client = where_are_you_ros2.where_are_you_action_client:main',
        ],
    },
)