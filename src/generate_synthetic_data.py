import random
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

random.seed(42)

# Per category: list of (subject, [description variants], intended_priority).
# Multiple phrasings give the classifier a rich vocabulary so it generalizes
# to novel wording instead of memorizing single sentences.
CATEGORY_TEMPLATES = {
    "Billing": [
        ("Incorrect charge", [
            "I was charged twice for my monthly subscription and the extra payment was never refunded to my card.",
            "My account got billed two times this month and nobody returned the duplicate money.",
            "They took the payment twice from my bank account and the second one was not credited back.",
        ], "High"),
        ("Double billing", [
            "My latest invoice shows a duplicate charge of the same amount and I need the second one reversed.",
            "The receipt I got lists the same line item twice and I want it removed.",
            "I can see two identical payments on my statement that I did not authorize.",
        ], "High"),
        ("Refund not received", [
            "I requested a refund three weeks ago but the money still has not been credited back to my account.",
            "It has been almost a month since I asked for my money back and nothing arrived.",
            "My reimbursement was approved but the funds never hit my bank account.",
        ], "High"),
        ("Unexpected fee", [
            "There is an unexpected service fee on my bill this month that I never agreed to. Please remove it.",
            "I was charged a strange convenience fee that was not mentioned when I signed up.",
            "An extra administrative charge showed up on my invoice which I never accepted.",
        ], "High"),
        ("Statement mismatch", [
            "The amount on my invoice does not match the plan I signed up for and I want it corrected.",
            "My paper bill shows a higher total than what our contract states.",
            "The final price on my statement is not the one we agreed to on the phone.",
        ], "Medium"),
        ("Duplicate payment", [
            "I noticed my card was debited twice for a single purchase and I would like the duplicate charge refunded.",
            "A single order ended up costing me double on my credit card and I need one charge eliminated.",
            "My checking account shows the same transaction repeated and it should only be once.",
        ], "High"),
        ("Automatic renewal charge", [
            "My subscription renewed automatically and charged me even though I wanted to downgrade. Refund please.",
            "The plan renewed on its own and took money from me although I asked to switch tiers earlier.",
            "I told support I was leaving the plan but I got billed for another cycle anyway.",
        ], "Medium"),
        ("Wrong tax", [
            "I was charged sales tax on an order that should be tax exempt. Please fix the invoice.",
            "My company order included tax even though we have a tax certificate on file.",
            "Sales tax was added to my purchase by mistake because our exemption is registered.",
        ], "Medium"),
    ],
    "Technical Issue": [
        ("Application crash", [
            "The application crashes every time I try to export the report and I lose all of my work.",
            "The program keeps closing unexpectedly and I cannot save my progress.",
            "Every time I open the dashboard the software shuts down on its own.",
        ], "Critical"),
        ("Login failure", [
            "I cannot log in to my account because the page keeps showing a server error after I enter my password.",
            "Signing in fails and the portal returns a gateway error every attempt.",
            "The login screen gives me an internal error before I even finish typing my credentials.",
        ], "High"),
        ("Feature broken", [
            "The export feature is broken and the button does not respond when I click it.",
            "Clicking download does nothing because the button is unresponsive and greyed out.",
            "The reporting function stopped working and pressing export gives no output.",
        ], "High"),
        ("Sync error", [
            "My data will not sync between my laptop and phone and I keep seeing an unknown error.",
            "The app fails to synchronize files across my devices and throws a warning.",
            "Changes on one device never show up on the other because syncing is interrupted.",
        ], "Medium"),
        ("App freezing", [
            "The app freezes and becomes unresponsive whenever I open the settings screen.",
            "The interface locks up completely as soon as I click on preferences.",
            "The program hangs for minutes when I try to change the configuration.",
        ], "High"),
        ("System down", [
            "Our company portal is completely down and none of the staff can access it right now.",
            "The whole website is unreachable and every employee is locked out of the workspace.",
            "We have a total outage and users inside the office cannot reach any service.",
        ], "Critical"),
        ("Error on startup", [
            "I get a fatal error every time the program starts and it closes immediately.",
            "Launching the tool immediately shows a red error and the window disappears.",
            "The software refuses to start and prints an exception on the console.",
        ], "High"),
        ("Network not working", [
            "My device lost connection to the office network and I cannot reconnect no matter what I try.",
            "The wifi drops every few minutes and the laptop never reconnects automatically.",
            "I cannot join the corporate network and the ethernet cable shows no connection.",
        ], "Medium"),
        ("Outage", [
            "Our production environment is offline and clients cannot reach any service right now.",
            "There is a total outage and customers are unable to access our platform.",
            "All of our servers are unreachable right now, this is a full service disruption.",
        ], "Critical"),
        ("Data loss", [
            "Important data disappeared from the system and we need it restored immediately.",
            "Files were deleted unexpectedly from the shared drive and this is urgent.",
            "Records vanished from our database and we have an emergency recovery request.",
        ], "Critical"),
    ],
    "Product Inquiry": [
        ("Feature question", [
            "I would like to know if the product supports multi-user collaboration for a team of ten.",
            "Can several people edit the same project at once with our plan?",
            "Does the tool have team sharing and co-editing features built in?",
        ], "Low"),
        ("Compatibility question", [
            "Can you confirm the software is compatible with the older version of our operating system?",
            "Will the application run on the operating system we are still using in our office?",
            "Is there any issue installing the product on our current infrastructure?",
        ], "Low"),
        ("Pricing details", [
            "What exactly is included in the premium pricing plan compared to the free tier?",
            "Could you list the differences between our basic and advanced packages?",
            "I want to know what extra benefits come with the paid subscription.",
        ], "Medium"),
        ("Upgrade process", [
            "How does the trial to paid upgrade work and will my existing data be preserved?",
            "If I move from the free version to a paying plan, do I keep all my stored projects?",
            "What is the process for converting our trial into a full enterprise account?",
        ], "Medium"),
        ("System requirements", [
            "Please share the minimum system requirements needed to run the application smoothly.",
            "What hardware and memory does the software need on an average workstation?",
            "Can you tell me the recommended specifications before we install it company wide?",
        ], "Low"),
        ("Integration options", [
            "Does this product integrate with our existing customer relationship management tool?",
            "Can we connect the platform with our current ticketing and analytics systems?",
            "What integrations are supported out of the box with third party apps?",
        ], "Low"),
        ("License question", [
            "What happens to my license if I need to move it to a different device?",
            "Can I transfer my activation key to a new laptop when this one is replaced?",
            "Does the license allow installation on more than one machine at the same time?",
        ], "Low"),
        ("Security features", [
            "Could you explain the security and encryption features available in the platform?",
            "What data protection and access control options does the enterprise offer?",
            "Tell me about the authentication and audit logging capabilities of the tool.",
        ], "Medium"),
    ],
    "Account Management": [
        ("Update profile", [
            "I need to update the email address and phone number associated with my account.",
            "Please change the contact details stored on my profile.",
            "My personal information has changed and I want it reflected in the system.",
        ], "Low"),
        ("Restore account", [
            "I accidentally closed my account and need help getting it restored with my data.",
            "My workspace was deleted by mistake and I want it brought back with everything inside.",
            "Can you recover my account and all the projects attached to it?",
        ], "Medium"),
        ("Add team member", [
            "I want to add several new members to our organization workspace with editor permissions.",
            "Please invite a few colleagues to join our project space.",
            "We hired new staff and they need accounts in the shared workspace.",
        ], "Medium"),
        ("Reset password", [
            "I forgot my password and the reset link never arrives in my inbox.",
            "The password recovery email is not being sent so I cannot regain access.",
            "I am unable to reset my login because no verification email shows up.",
        ], "Medium"),
        ("Change plan", [
            "Please help me downgrade my current plan to the basic tier before the next billing cycle.",
            "We want to move to a smaller plan to reduce our monthly spending.",
            "Is it possible to switch our subscription to the cheaper package from now?",
        ], "Medium"),
        ("Manage roles", [
            "I need to change the role of a user in our workspace from viewer to admin.",
            "One of our users should get elevated permissions to manage projects.",
            "Please grant administrator rights to my colleague on the main workspace.",
        ], "Low"),
        ("Access request", [
            "One of our contractors needs temporary access to the project folder for two weeks.",
            "Give a visiting engineer read access to the repository until next month.",
            "We have an external vendor who requires limited access to our files.",
        ], "Medium"),
        ("Notification settings", [
            "I want to change which email notifications I receive for project activity.",
            "Please stop sending me daily summaries for every minor update in the workspace.",
            "Adjust my alert preferences so I only hear about critical changes.",
        ], "Low"),
    ],
    "Cancellation": [
        ("Cancel subscription", [
            "I would like to cancel my subscription completely and stop all future charges.",
            "Please end my plan right away and ensure no more payments are taken.",
            "I want to terminate my account and close the contract immediately.",
        ], "Medium"),
        ("Stop auto renewal", [
            "Please turn off auto renewal so I am not charged automatically next month.",
            "Disable the automatic renewal on my plan before the next cycle begins.",
            "I do not want the service to renew itself when the period is over.",
        ], "Medium"),
        ("Delete account", [
            "I want to permanently delete my account and remove all of my personal data.",
            "Please erase my profile and every record associated with it from your systems.",
            "Remove my account and all stored information for good.",
        ], "High"),
        ("End of contract", [
            "Our company contract is ending and we need to cancel our yearly plan.",
            "The annual agreement expires soon and we will not be renewing it.",
            "We are closing the long term deal and would like to terminate the subscription.",
        ], "Low"),
        ("Remove payment method", [
            "I need to delete my saved credit card before the next billing cycle.",
            "Remove my stored card details so nothing can be charged in the future.",
            "Please delete the payment method attached to my account.",
        ], "Medium"),
        ("Cancel pending order", [
            "I placed an order by mistake and would like to cancel it before it ships.",
            "Please void my recent purchase because it was created accidentally.",
            "I need to stop an order that has not been dispatched yet.",
        ], "High"),
        ("Terminate service", [
            "We are switching providers and need to terminate all services at the end of the month.",
            "Cancel every service we use as we are moving to a different vendor.",
            "Shut down our account and services once the current period finishes.",
        ], "Medium"),
        ("Close workspaces", [
            "Please close our organization workspace and all sub-workspaces as we are shutting down.",
            "Our company is closing and every workspace on our billing should be removed.",
            "Terminate all the team spaces associated with our account.",
        ], "Medium"),
    ],
}

PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]
NEIGHBOR_FLIP_PROB = 0.06

URGENT_APPEND = [
    "This is urgent, please help immediately.",
    "We need a resolution asap.",
    "Please fix this right away.",
    "Nobody can work until this is resolved.",
    "This is blocking our whole team right now.",
]

CALM_APPEND = [
    "No rush, whenever someone has time.",
    "This is not urgent but please look into it.",
    "Whenever convenient, thanks.",
    "There is no deadline, just wanted to ask.",
]

ROWS = 6000


def flip_priority(priority):
    """With small probability, move one rung up/down: simulates ~human label noise."""
    if random.random() >= NEIGHBOR_FLIP_PROB:
        return priority
    i = PRIORITY_ORDER.index(priority)
    i += random.choice([-1, 1])
    i = max(0, min(len(PRIORITY_ORDER) - 1, i))
    return PRIORITY_ORDER[i]


def augment(description, priority):
    if priority in ("High", "Critical") and random.random() < 0.45:
        return description + " " + random.choice(URGENT_APPEND)
    if priority == "Low" and random.random() < 0.40:
        return description + " " + random.choice(CALM_APPEND)
    if priority == "Medium" and random.random() < 0.20:
        return description + " " + random.choice(CALM_APPEND)
    return description


def generate():
    data = []
    categories = list(CATEGORY_TEMPLATES.keys())
    while len(data) < ROWS:
        category = random.choice(categories)
        subject, descriptions, base_priority = random.choice(CATEGORY_TEMPLATES[category])
        description = random.choice(descriptions)
        description = augment(description, base_priority)
        priority = flip_priority(base_priority)
        data.append({
            "Ticket Subject": subject,
            "Ticket Description": description,
            "Ticket Type": category,
            "Ticket Priority": priority,
        })
    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/customer_support_tickets.csv", index=False)
    print(f"Generated {len(df)} tickets -> data/customer_support_tickets.csv")
    print(df["Ticket Type"].value_counts())
    print(df["Ticket Priority"].value_counts())