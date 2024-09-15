# Prompt Generation
def generate_prompt_training(data_point):
    return f"""
    Task Description:
    Review user reviews for mobile applications based on their content, sentiment, and ratings. Utilize the definitions provided to classify each review into the appropriate category.
    Definitions for Classification:
    Bug Reports:
    Definition: Bug reports are user comments that identify issues with the app, such as crashes, incorrect behavior, or performance problems. These reviews specifically highlight problems that affect the app's functionality and suggest a need for corrective action.
    Feature Requests:
    Definition: Feature requests are suggestions by users for new features or enhancements in future app updates. These can include requests for features seen in other apps, additions to content, or ideas to modify existing features to enhance user interaction and satisfaction.
    User Experience:
    Definition: User experience reviews provide detailed narratives focusing on specific app features and their effectiveness in real scenarios. They offer insights into the app’s usability, functionality, and overall satisfaction, often serving as informal documentation of user needs and app performance.
    Differentiating Tip: Prioritize reviews that give detailed explanations of the app's features and their practical impact on the user.
    Ratings:
    Definition: Ratings are brief textual comments that reflect the app's numeric star rating, primarily indicating overall user satisfaction or dissatisfaction. These reviews are succinct, focusing on expressing a general sentiment without detailed justification.
    Differentiating Tip: Focus on reviews that lack detailed discussion of specific features or user experiences, and instead provide general expressions of approval or disapproval.
    Instructions to the Language Model:
    Review Processing: Carefully read the provided app review and its star rating.
    And Classify the review into one of the following categories: "Bug", "Feature", "UserExperience", or "Rating".
    Output Format: Provide the classification results in the following JSON format:
    {{
        "Class": "<predition>"
    }}
    Review and Star Rating to Classify:
    Review: 
    User Review : {data_point["content"]}        
    User Rating : {str(data_point["score"])} out of 5

    Prediction: 
    {{
        "Class": "{data_point["llm_annotation"]}"
    }}
"""
def generate_prompt_training_with_Explanation(data_point):
    return f"""
    Task Description:
    Review user reviews for mobile applications based on their content, sentiment, and ratings. Utilize the definitions provided to classify each review into the appropriate category.
    Definitions for Classification:
    Bug Reports:
    Definition: Bug reports are user comments that identify issues with the app, such as crashes, incorrect behavior, or performance problems. These reviews specifically highlight problems that affect the app's functionality and suggest a need for corrective action.
    Feature Requests:
    Definition: Feature requests are suggestions by users for new features or enhancements in future app updates. These can include requests for features seen in other apps, additions to content, or ideas to modify existing features to enhance user interaction and satisfaction.
    User Experience:
    Definition: User experience reviews provide detailed narratives focusing on specific app features and their effectiveness in real scenarios. They offer insights into the app’s usability, functionality, and overall satisfaction, often serving as informal documentation of user needs and app performance.
    Differentiating Tip: Prioritize reviews that give detailed explanations of the app's features and their practical impact on the user.
    Ratings:
    Definition: Ratings are brief textual comments that reflect the app's numeric star rating, primarily indicating overall user satisfaction or dissatisfaction. These reviews are succinct, focusing on expressing a general sentiment without detailed justification.
    Differentiating Tip: Focus on reviews that lack detailed discussion of specific features or user experiences, and instead provide general expressions of approval or disapproval.
    Instructions to the Language Model:
    Review Processing: Carefully read the provided app review and its star rating.
    Give a brief explanation of the classification decision made for the review and Classify the review into one of the following categories: "Bug", "Feature", "UserExperience", or "Rating".
    Output Format: Provide the classification results in the following JSON format:
    {{
        "Explanation": "<explanation>",
        "Class": "<predition>"
    }}
    Review and Star Rating to Classify:
    Review: 
    User Review : {data_point["content"]}        
    User Rating : {str(data_point["score"])} out of 5

    Prediction: 
    {{
        "Explanation": "{data_point["llm_explanation"]}",
        "Class": "{data_point["llm_annotation"]}"
    }}
"""

def generate_prompt_validation(data_point):
    return f"""
    Task Description:
    Review user reviews for mobile applications based on their content, sentiment, and ratings. Utilize the definitions provided to classify each review into the appropriate category.
    Definitions for Classification:
    Bug Reports:
    Definition: Bug reports are user comments that identify issues with the app, such as crashes, incorrect behavior, or performance problems. These reviews specifically highlight problems that affect the app's functionality and suggest a need for corrective action.
    Feature Requests:
    Definition: Feature requests are suggestions by users for new features or enhancements in future app updates. These can include requests for features seen in other apps, additions to content, or ideas to modify existing features to enhance user interaction and satisfaction.
    User Experience:
    Definition: User experience reviews provide detailed narratives focusing on specific app features and their effectiveness in real scenarios. They offer insights into the app’s usability, functionality, and overall satisfaction, often serving as informal documentation of user needs and app performance.
    Differentiating Tip: Prioritize reviews that give detailed explanations of the app's features and their practical impact on the user.
    Ratings:
    Definition: Ratings are brief textual comments that reflect the app's numeric star rating, primarily indicating overall user satisfaction or dissatisfaction. These reviews are succinct, focusing on expressing a general sentiment without detailed justification.
    Differentiating Tip: Focus on reviews that lack detailed discussion of specific features or user experiences, and instead provide general expressions of approval or disapproval.
    Instructions to the Language Model:
    Review Processing: Carefully read the provided app review and its star rating.
    And Classify the review into one of the following categories: "Bug", "Feature", "UserExperience", or "Rating".
    Output Format: Provide the classification results in the following JSON format:
    {{
        "Class": "<predition>"
    }}
    Review and Star Rating to Classify:
    Review: 
    User Review : {data_point["comment"]}        
    User Rating : {str(data_point["rating"])} out of 5
    
    Prediction: 
"""
def generate_prompt_validation_with_Explanation(data_point):
    return f"""
    Task Description:
    Review user reviews for mobile applications based on their content, sentiment, and ratings. Utilize the definitions provided to classify each review into the appropriate category.
    Definitions for Classification:
    Bug Reports:
    Definition: Bug reports are user comments that identify issues with the app, such as crashes, incorrect behavior, or performance problems. These reviews specifically highlight problems that affect the app's functionality and suggest a need for corrective action.
    Feature Requests:
    Definition: Feature requests are suggestions by users for new features or enhancements in future app updates. These can include requests for features seen in other apps, additions to content, or ideas to modify existing features to enhance user interaction and satisfaction.
    User Experience:
    Definition: User experience reviews provide detailed narratives focusing on specific app features and their effectiveness in real scenarios. They offer insights into the app’s usability, functionality, and overall satisfaction, often serving as informal documentation of user needs and app performance.
    Differentiating Tip: Prioritize reviews that give detailed explanations of the app's features and their practical impact on the user.
    Ratings:
    Definition: Ratings are brief textual comments that reflect the app's numeric star rating, primarily indicating overall user satisfaction or dissatisfaction. These reviews are succinct, focusing on expressing a general sentiment without detailed justification.
    Differentiating Tip: Focus on reviews that lack detailed discussion of specific features or user experiences, and instead provide general expressions of approval or disapproval.
    Instructions to the Language Model:
    Review Processing: Carefully read the provided app review and its star rating.
    Give a brief explanation of the classification decision made for the review and Classify the review into one of the following categories: "Bug", "Feature", "UserExperience", or "Rating".
    Output Format: Provide the classification results in the following JSON format:
    {{
        "Explanation": "<explanation>",
        "Class": "<predition>"
    }}
    Review and Star Rating to Classify:
    Review: 
    User Review : {data_point["comment"]}        
    User Rating : {str(data_point["rating"])} out of 5
    
    Prediction: 
"""