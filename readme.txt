# Combined Flask Application

This project merges two separate Flask applications into one, with different API endpoints for evaluating SMART criteria and analyzing similarity scores. The project uses transformers, NLTK, and sentence-transformers libraries.

## API Endpoints

1. **http://localhost:5000/evaluate**: Evaluate SMART criteria.
2. **http://localhost:5000/analyze**: Analyze similarity scores.

## Requirements

- Docker
- Python 3.9

## Setup Instructions

1. **Build the Docker image:**

    ```sh
    docker build -t combined-app .
    ```

2. **Run the Docker container:**

    ```sh
    docker run -p 5000:5000 combined-app
    ```

## Endpoints Usage

### /evaluate

**Request:**

- **Method:** POST
- **Content-Type:** application/json
- **Body:**
    ```json
    {
      "artifactName": "ProjectCharter",
      "elements": {
        "Project Purpose or Justification": "UC Berkeley does not use energy as efficiently or as wisely as it could, leading to wastage. The project aims to emphasize individual actions through campus outreach to reduce campus and auxiliary energy usage, provide relevant information to stakeholders, and achieve a high participation rate.",
        "Objectives and Scope": "The project aims to develop, launch, and maintain a campus-wide campaign on energy reduction, accompanied by an incentive program for Operating Units. It includes the development of a campaign name, visual identity, website, marketing plan, competitions, energy audits, resources, communication with stakeholders, and coordination with the OE Procurement and IT teams.",
        "High-Level Requirements": "The project requires installation of proposed meters and related software at the beginning of the outreach campaign, demand for energy audits and resources to be met, finalization of decisions from procurement and IT teams affecting energy saving potential, and the buy-in and participation of Operating Units for estimated energy savings to be realized.",
        "Project Description": "The Marketing and Outreach project aims to address the inefficiency and wastage of energy at UC Berkeley by emphasizing individual actions through campus outreach and a campus-wide campaign on energy reduction.",
        "Deliverables": "Deliverables include the development and launch of the campus-wide campaign, creation of a website, energy usage records, visual identity, marketing materials, energy audits, and the roll-out of the campaign to all buildings.",
        "Success Criteria": {
          "1": "Reduced total campus and auxiliary energy usage",
          "2": "Relevant information provided to stakeholders",
          "3": "High Operating Unit participation rate"
        },
        "Start and End Dates": "Start date: Summer-Winter 2011; End date: On-going",
        "Key Dates and Milestones": {
          "Implementation team fully staffed": "July 2011; Summer-Winter 2011",
          "Formalized working relationship with Energy Office": "Summer 2011",
          "Marketing plan and materials developed": "By initial launch: Oct/Nov 2011",
          "Energy audits conducted (with Energy Office)": "Initial set of buildings: by initial launch (Oct/Nov 2011); on-going",
          "Initial launch": "Oct/Nov 2011",
          "Campaign rolled-out to all buildings": "Nov 2011 – on-going"
        },
        "Phases of Work": "Details may change upon the completion of the final workplan",
        "Constraints": "Constraints include assumptions on the implementation of the Energy Office and Incentive Program, potential greater demand for energy audits and resources than supply, finalization of decisions from procurement and IT teams, and the buy-in and participation of Operating Units.",
        "Assumptions": "Assumptions include the implementation of the Energy Office and Incentive Program, availability of resources for energy audits, timely finalization of decisions from procurement and IT teams, and the buy-in and participation of Operating Units.",
        "High-Level Project Risks": {
          "Risk": "Participation may be too low, program may be perceived as not fully successful, and financial savings may not be a sufficient motivator",
          "Mitigation Strategy": "Involvement of Operating Units, assessments and presentation of metrics, and an extensive marketing and outreach campaign"
        },
        "Budget Summary": "The project will require $510,000 in OE funding, with expected run-rate savings of $700,000.",
        "Stakeholder List": "Not available in the provided document",
        "Project Organization": "Not available in the provided document",
        "Approval Requirements": "The Project Sponsor signature indicates approval of the Project Charter",
        "Change Management Process": "Scope additions or changes will require a scope change request and formal approval by the Project Sponsor.",
        "Communication Plan": "Not available in the provided document"
      }
    }
    ```

### /analyze

**Request:**

- **Method:** POST
- **Content-Type:** application/json
- **Body:**
    ```json
    {
      "artifacts": [
        {
          "artifactName": "Charter",
          "GROKS": {
            "Goals": {
              "Secure Funding": "To secure $50,000 in funding to establish Donny's Food Truck restaurant in Small Town.",
              "Establish Business Venture": "To provide the community with a diverse and convenient dining option while creating a successful and sustainable business venture.",
              "Achieve Footfall": "To achieve a daily footfall of at least 100 customers within the first month.",
              "Receive Positive Reviews": "To receive positive customer reviews and feedback on food quality and service.",
              "Break Even": "To break even within six months of operation.",
              "Establish Partnerships": "To establish partnerships with local events and businesses for regular appearances."
            },
            "Resources": {
              "Food Truck": "High-quality food truck equipped with kitchen essentials.",
              "Marketing Materials": "Banners, flyers, and social media campaigns.",
              "Staff": "Skilled and friendly staff members trained in food safety and customer service.",
              "Menu Options": "Diverse menu options catering to different tastes and dietary preferences."
            },
            "KPIs": {
              "Footfall": "Achieve a daily footfall of at least 100 customers within the first month.",
              "Customer Feedback": "Receive positive customer reviews and feedback on food quality and service.",
              "Financial": "Break even within six months of operation."
            }
          }
        },
        {
          "artifactName": "Business Plan",
          "GROKS": {
            "Goals": {
              "Secure Funding": "To secure $50,000 in funding to establish Donny's Food Truck restaurant in Small Town.",
              "Establish Business Venture": "To provide the community with a diverse and convenient dining option while creating a successful and sustainable business venture.",
              "Achieve Footfall": "To achieve a daily footfall of at least 100 customers within the first month.",
              "Receive Positive Reviews": "To receive positive customer reviews and feedback on food quality and service.",
              "Break Even": "To break even within six months of operation.",
              "Establish Partnerships": "To establish partnerships with local events and businesses for regular appearances."
            },
            "Resources": {
              "Food Truck": "High-quality food truck equipped with kitchen essentials.",
              "Marketing Materials": "Banners, flyers, and social media campaigns.",
              "Staff": "Skilled and friendly staff members trained in food safety and customer service.",
              "Menu Options": "Diverse menu options catering to different tastes and dietary preferences."
            },
            "KPIs": {
              "Footfall": "Achieve a daily footfall of at least 100 customers within the first month.",
              "Customer Feedback": "Receive positive customer reviews and feedback on food quality and service.",
              "Financial": "Break even within six months of operation."
            }
          }
        }
      ]
    }
    ```

## Dependencies

All dependencies are listed in the `requirements.txt` file. The main dependencies are:

- Flask==2.3.2
- nltk
- sentence-transformers
- pandas
- transformers==4.29.2
- torch==2.0.1
- numpy<2

## Notes

- Ensure Docker is installed and running on your system.
- The application will be available at `http://localhost:5000` after running the Docker container.
