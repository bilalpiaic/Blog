# How I Built an Automated Lead Generation System That Finds 50+ B2B Prospects Daily

> **Author**: Eason Meng
>
> **Follow me on**: [LinkedIn](https://www.linkedin.com/in/yusen-meng-4946b6135/) | [Twitter](https://x.com/luluyuyusese)
>
> **Tip**: For a better reading experience, please visit the [full article on GitHub](https://github.com/YukoOshima/Blog/blob/main/articles/outbound_leads.md).
>
> **Resources**: Firecrawl n8n Integration Guide: [https://www.firecrawl.dev/blog/firecrawl-n8n-web-automation](https://www.firecrawl.dev/blog/firecrawl-n8n-web-automation)

## The Problem I Was Trying to Solve

Like most B2B founders, I was spending hours every day manually searching for potential customers. I'd open LinkedIn, Google random industry keywords, visit company websites one by one, and try to figure out if they were a good fit for our product. By the end of the day, I'd have maybe 10-15 prospects, and honestly, half of them weren't even that great.

The math was depressing. If I could only find 10 prospects a day, and my conversion rate was around 2%, I was looking at one customer every five days. That's roughly 6 customers per month. For a business targeting $30K+ monthly revenue, that just wasn't going to cut it.

That's when I decided to build an automated system using n8n, Firecrawl, and Claude AI. Now, this system finds 50+ qualified prospects daily with minimal effort from me. More importantly, because the AI does the heavy lifting on qualification, my conversion rate has jumped to around 8%.

## How the System Actually Works

The beauty of this setup is that it mimics exactly what I was doing manually, but at 10x the speed and with much better analysis. Here's how it flows:

Instead of me typing keywords into Google, the system does intelligent searches using combinations like "SaaS companies + recently funded" or "marketing agencies + hiring developers." It's not just throwing random searches either - it's strategic about finding companies that match our ideal customer profile.

Once Firecrawl finds these companies, it doesn't just grab their homepage. It intelligently crawls through their About pages, team sections, recent blog posts, and even their pricing pages. This gives me a complete picture of who they are, what they do, and how much they might be willing to spend.

```mermaid
graph LR
    A[Smart Search] --> B[Website Crawling]
    B --> C[AI Analysis]
    C --> D[Personalized Insights]
    D --> E[Quality Scoring]
    E --> F[Team Notifications]

    style A fill:#e1f5fe,stroke:#01579b
    style B fill:#e8f5e9,stroke:#2e7d32
    style C fill:#fff3e0,stroke:#ef6c00
    style D fill:#f3e5f5,stroke:#7b1fa2
    style F fill:#e0f2f1,stroke:#00695c
```

Here's where Claude AI comes in and does something I could never do manually at scale. For each company, it analyzes all the collected information and tells me:

What problems this company likely has that my product could solve. For example, if I see they're hiring a lot of developers but their careers page mentions "fast-paced environment," Claude might identify scaling challenges.

How good of a fit they are based on industry, company size, and tech stack. A 50-person fintech company using React might score a 9/10 for our developer tools, while a 5-person local restaurant might score a 2/10.

The best angle to approach them with. Instead of generic outreach, I get specific talking points like "They just raised Series A and are likely struggling with technical debt as they scale."

What their budget range probably looks like. A company that just raised $10M is going to have different purchasing power than a bootstrapped startup.

## The Magic is in the Personalization

The game-changer is that every prospect comes with a personalized value proposition. When I reach out, I'm not sending generic "Hey, want to see our product?" emails. I'm sending messages like:

"Hi [Name], I noticed you recently expanded to 15 engineers and are hiring aggressively. Most companies at your stage struggle with code review bottlenecks that slow down deployments. Our tool has helped similar fintech companies reduce review time by 60%. Worth a quick chat?"

That level of personalization at scale is what drives the 8% conversion rate.

## What This Looks Like in Practice

Let me walk you through a real example. Say I'm selling developer tools to fintech companies.

The system searches for "fintech startups hiring developers" and finds 50 companies. Firecrawl visits each website and pulls information about their tech stack, team size, recent funding, and job postings.

Claude analyzes all this data and identifies that TechCorp (fake name) is a Series A fintech with 25 employees, using React and Node.js, recently posted 5 developer jobs, and their CEO just wrote a blog post about "scaling challenges."

Based on this analysis, Claude generates a personalized approach: "This company is in rapid growth mode, likely struggling with code quality and deployment speed. Approach angle: help them maintain code quality while scaling their team. Estimated budget: $5K-15K/month. Best contact: Engineering Manager (found on their team page)."

When this lands in my Slack, I have everything I need to write a compelling, personalized outreach message.

## Taking It to the Next Level

Once you have the basic system running, there are some advanced features that can really multiply your results.

The first is competitive intelligence. Instead of just finding random prospects, you can specifically target your competitors' customers. The system can search for companies mentioning competitor tools on their websites or job postings, then analyze whether they might be good candidates for switching.

I've also added social media monitoring that tracks LinkedIn company updates, recent hires in key positions, and funding announcements. When a company gets new funding or hires a new CTO, that's often the perfect time to reach out with your product.

The email automation piece is where things get really interesting. The system doesn't just find prospects and analyze them - it can actually generate the first outreach email and set up follow-up sequences. Of course, I still review everything before it goes out, but having a personalized draft ready saves tons of time.

## The Numbers Don't Lie

Before building this system, I was stuck in the manual prospecting grind. Every day, I'd spend 3-4 hours researching companies and end up with maybe 10 prospects. My conversion rate was around 2%, so I was getting roughly one new customer every five days.

Now, the system runs in the background and delivers 50+ qualified prospects daily. Because Claude does such a good job with the initial qualification and personalization, my conversion rate has jumped to 8%. Instead of 6 customers per month, I'm looking at 120+ qualified opportunities.

The time savings alone are incredible. What used to take me 4 hours now takes about 30 minutes of my time each morning to review the prospects and customize the final outreach messages.

## How to Actually Build This

If you want to set this up yourself, here's the technical overview. The whole thing runs on n8n, which is like Zapier but more powerful for complex workflows.

For the web scraping, I use Firecrawl because it handles JavaScript-heavy sites better than traditional scrapers and has built-in n8n integration. You can set it up with either HTTP requests (which I recommend) or their community node if you're self-hosting n8n.

Here's a simplified version of the Firecrawl API call:

```json
{
  "method": "POST",
  "url": "https://api.firecrawl.dev/v0/scrape",
  "headers": {
    "Authorization": "Bearer fc-YOUR_API_KEY",
    "Content-Type": "application/json"
  },
  "body": {
    "url": "{{ $json.company_url }}",
    "extractorOptions": {
      "extractionPrompt": "Extract company information including: company name, industry, employee count, recent news, contact information, and technology mentions."
    }
  }
}
```

For the AI analysis, I send all the scraped data to Claude with a prompt like this:

```
Analyze this company and provide:
1. What business problems they likely have
2. How well our [product] would solve their problems (1-10 score)
3. Estimated budget range
4. Best approach for outreach
5. Personalized value proposition

Company Data: [scraped information]
```

The filtering and notification parts are just standard n8n nodes. I use the Filter node to only keep prospects that score above 7/10, then send the good ones to Slack with all the analysis included.

## Things to Keep in Mind

When you're building something like this, there are a few important considerations. First, be respectful with your scraping. Don't hammer websites with requests - add delays between calls and respect robots.txt files. Most sites won't mind reasonable scraping for business development, but you don't want to be the person who crashes someone's server.

Data privacy is another big one. Make sure you're compliant with GDPR, CCPA, and other regulations. I anonymize personal data where possible and have clear data retention policies.

The AI prompts are probably the most important part to get right. I spent weeks tweaking the Claude prompts to get consistent, high-quality analysis. Start simple and gradually add more criteria as you see what works.

## Why Claude AI Works Best

While you can technically use any AI model for the analysis part - GPT-4, Gemini, or even open-source models - I've tested most of them extensively and Claude consistently delivers the best results for this use case.

Claude excels at understanding nuanced business contexts and generating natural, personalized messaging that doesn't sound robotic. It's particularly good at identifying pain points from limited company information and creating compelling value propositions that actually resonate with prospects.

That said, the system is flexible. If you prefer a different model or want to experiment, you can easily swap out the API calls. Just keep in mind that you'll likely need to adjust your prompts significantly since each model has different strengths and quirks.

## Common Issues and How to Fix Them

The biggest problem I ran into early on was getting too many low-quality leads. The solution was being more specific with my search keywords and adding additional filters in the AI analysis step.

Rate limiting from websites is another common issue. The solution is to add random delays between requests and rotate through different search approaches.

If you're getting inconsistent results from Claude, it's usually a prompt problem. The more specific and structured your prompt, the more consistent the results.

## What's Next for This System

I'm constantly adding new features. The latest addition is integration with LinkedIn Sales Navigator to pull additional company information and find specific contact details.

I'm also experimenting with image analysis to understand company culture from their website photos and social media. It sounds weird, but you can actually learn a lot about a company's stage and priorities from how they present themselves visually.

The email automation is getting more sophisticated too. Instead of just generating one outreach email, the system now creates entire sequence campaigns with different angles and follow-up timing.

## Final Thoughts

Building this automated lead generation system has been one of the highest-impact projects I've worked on. It's transformed our sales process from a manual, time-consuming grind into a scalable, data-driven machine.

The best part is that it gets better over time. As I feed successful and unsuccessful outcomes back into the system, the AI gets better at identifying what makes a good prospect for our specific product.

If you're spending hours every week manually prospecting, I highly recommend building something like this. Start simple with just the basic search and scraping, then gradually add the AI analysis and personalization features.

The tools are all there and more accessible than ever. You just need to put in the time to set it up properly.