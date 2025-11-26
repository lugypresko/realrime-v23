# Sales Assistant Prompt Library
# These are the prompts that will be matched against client speech

PROMPTS = [
    # Objection Handling
    "Ask them what specific concerns they have about the pricing.",
    "Acknowledge their budget constraint and offer a scaled-down option.",
    "Ask if they've compared this to their current solution's total cost.",
    "Suggest a pilot program to prove ROI before full commitment.",
    
    # Discovery Questions
    "Ask about their current process and pain points.",
    "Find out who else is involved in the decision-making process.",
    "Ask what their timeline looks like for implementing a solution.",
    "Clarify what success would look like for them in 6 months.",
    
    # Value Reinforcement
    "Remind them of the ROI calculation you discussed earlier.",
    "Reference the case study from their industry you mentioned.",
    "Highlight the specific feature that solves their main pain point.",
    "Emphasize the time savings compared to their current workflow.",
    
    # Next Steps
    "Propose scheduling a technical demo with their team.",
    "Suggest sending over the proposal by end of day.",
    "Ask if they'd like to speak with a current customer as a reference.",
    "Offer to prepare a customized implementation timeline.",
    
    # Engagement Recovery
    "Ask if they need clarification on anything you've covered.",
    "Check if this still aligns with their priorities.",
    "Suggest taking a brief pause if they need to check something.",
    "Ask if there's someone else who should join this conversation.",
    
    # Closing Signals
    "Ask what would need to happen for them to move forward today.",
    "Propose specific next steps with dates.",
    "Check if there are any remaining blockers to address.",
]

if __name__ == "__main__":
    print(f"Prompt Library: {len(PROMPTS)} prompts")
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"{i:2d}. {prompt}")
