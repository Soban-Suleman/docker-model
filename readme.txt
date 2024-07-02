# Combined Flask Application

This project merges two separate Flask applications into one, with different API endpoints for evaluating SMART criteria and analyzing similarity scores. The project uses transformers, NLTK, and sentence-transformers libraries.

## API Endpoints

1. **http://localhost:5000/evaluate_smat_criteria**: Evaluate Feasibility / SMART criteria.
2. **http://localhost:5000/analyze**: Analyze Similarity/Alignment scores.

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

### /evaluate_smat_criteria

**Request:**

- **Method:** POST
- **Content-Type:** application/json
- **Body:**
    ```json
{
  "artifactName":"ProjectCharter",
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
}    ```

### /analyze

**Request:**

- **Method:** POST
- **Content-Type:** application/json
- **Body:**
    ```json
{
  "artifacts": [
    {
      "artifactName": "ProjectStrategy",
      "GOCKS": {
        "Goals": {
          "Sustainable Energy Solutions": "Develop innovative technologies to harness renewable energy sources, reduce dependency on fossil fuels, and meet regulatory standards for environmental sustainability.",
          "Social Needs": "Mitigate climate change and promote ecological balance."
        },
        "Objectives": {
          "Advanced Technologies": "Develop advanced renewable energy technologies that are efficient and cost-effective.",
          "Emissions Reduction": "Reduce greenhouse gas emissions by promoting the use of renewable energy.",
          "Energy Security": "Enhance energy security by diversifying energy sources.",
          "Technology Adoption": "Facilitate the adoption of renewable energy technologies in various sectors.",
          "Regulatory Compliance": "Meet regulatory requirements and support policy frameworks for sustainable development."
        },
        "Constraints": {
          "Budget": "Fixed budget of $10 million.",
          "Deadline": "Completion deadline of December 31, 2026.",
          "Equipment Availability": "Limited availability of specialized research equipment."
        },
        "KPIs": {
          "Project Milestones": "Achievement of project milestones such as completion of feasibility studies, prototype development, and testing phases.",
          "Budget Adherence": "Project completion within the allocated budget.",
          "Stakeholder Satisfaction": "Satisfaction of stakeholders with the project outcomes."
        },
        "Strategy": {
          "Collaborations": "Collaborating with academic institutions, industry partners, and regulatory bodies to ensure the project's alignment with technological and market trends.",
          "Hybrid Methodology": "Using a hybrid methodology combining Agile and Waterfall approaches to ensure flexibility and thorough documentation.",
          "Resource Management": "Efficiently managing human, financial, and material resources to support research and development activities.",
          "Risk Management": "Identifying potential risks and developing mitigation strategies, including contingency plans and regular risk reviews."
        }
      }
    },
    {
      "artifactName": "BusinessObjective",
      "GOCKS": {
        "Goals": {
          "Customer Happiness": "Improve customer happiness and comfort through high-quality, affordable food options.",
          "Sales Targets": "Achieve annual sales of $100,000 in the first year and $212,000 by the end of the third year."
        },
        "Objectives": {
          "Funding": "Secure $50,000 for purchase, marketing, and staffing needs.",
          "Expansion": "Expand services to multiple locations and offer diverse food options."
        },
        "Constraints": {
          "Financial": "Limited startup capital of $50,000, with a 3-year repayment plan at 5% interest.",
          "Market Competition": "Compete with local food vendors offering similar products at higher prices."
        },
        "KPIs": {
          "Sales Growth": "Achieve projected sales of $100,000 in the first year and $212,000 by the third year.",
          "Customer Reach": "Target customers aged 19 to 35 years through various marketing strategies."
        },
        "Strategy": {
          "Marketing": "Utilize social media, flyers, coupons, word of mouth, and partnerships with local vendors.",
          "Operations": "Ensure operational efficiency through quality control, resource utilization, and regulatory compliance."
        }
      }
    }
  ]
} 

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
