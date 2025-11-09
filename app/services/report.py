import asyncio
import sys
import os
from datetime import datetime
from typing import List, Dict
from app.models import ReportSection, ResearchReportResponse
from app.llm import llm, load_company_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from app.report_prompts import (
    article_prompt, 
    create_section_prompt, 
    REPORT_SECTIONS, 
    reduce_prompt
)


# --------------------------
# Map-Reduce Report Generation
# --------------------------

async def generate_section(
    company_name: str,
    retriever,
    section_config: Dict[str, str]
) -> ReportSection:
    """
    MAP Phase: Generate a single section of the report.
    
    Args:
        company_name: Company name
        retriever: LangChain retriever
        section_config: Dictionary with 'title', 'focus', and 'query_template'
    
    Returns:
        ReportSection object
    """
    try:
        section_title = section_config["title"]
        section_focus = section_config["focus"]
        query_template = section_config["query_template"]
        
        # Create section-specific prompt
        section_prompt = create_section_prompt(section_title, section_focus)
        
        # Create document chain with section prompt
        document_chain = create_stuff_documents_chain(llm, section_prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Generate query for this section
        query = f"{company_name} {query_template}"
        
        # Generate the section
        result = await retrieval_chain.ainvoke({"input": query})
        
        # Extract results
        section_content = result.get("answer", "")
        retrieved_docs = result.get("context", [])
        sources = [doc.metadata.get("source", "unknown") for doc in retrieved_docs]
        
        return ReportSection(
            title=section_title,
            content=section_content,
            sources_used=sources
        )
    
    except Exception as e:
        print(f"⚠️ Error generating section '{section_config.get('title', 'Unknown')}': {e}")
        # Return error section
        return ReportSection(
            title=section_config.get("title", "Unknown Section"),
            content=f"Error generating section: {str(e)}",
            sources_used=[]
        )


async def generate_all_sections(
    company_name: str,
    retriever,
    sections_config: List[Dict[str, str]]
) -> List[ReportSection]:
    """
    MAP Phase: Generate all sections sequentially (one by one).
    
    Args:
        company_name: Company name
        retriever: LangChain retriever
        sections_config: List of section configuration dictionaries
    
    Returns:
        List of ReportSection objects
    """
    print(f"📊 Generating {len(sections_config)} sections sequentially (MAP phase)...")
    
    valid_sections = []
    
    # Generate sections one by one to avoid timeouts
    for i, section_config in enumerate(sections_config, 1):
        section_title = section_config.get("title", "Unknown Section")
        print(f"  [{i}/{len(sections_config)}] Generating: {section_title}...")
        
        try:
            section = await generate_section(company_name, retriever, section_config)
            
            if section.content and not section.content.startswith("Error generating"):
                valid_sections.append(section)
                print(f"  ✅ {section.title} - completed")
            else:
                print(f"  ⚠️ {section.title} - skipped (empty or error)")
        
        except Exception as e:
            print(f"  ❌ {section_title} - failed: {e}")
            # Continue with next section even if one fails
            continue
    
    print(f"  📝 Successfully generated {len(valid_sections)}/{len(sections_config)} sections")
    
    return valid_sections


async def combine_sections(
    company_name: str,
    sections: List[ReportSection]
) -> str:
    """
    REDUCE Phase: Combine all sections into a cohesive final article.
    
    Args:
        company_name: Company name
        sections: List of ReportSection objects
    
    Returns:
        Combined article text
    """
    if not sections:
        return "No sections were generated successfully."
    
    print(f"🔄 Combining {len(sections)} sections into final article (REDUCE phase)...")
    
    # Format sections for the reduce prompt
    sections_text = "\n\n".join([
        f"## {section.title}\n\n{section.content}"
        for section in sections
    ])
    
    # Try LCEL (LangChain Expression Language) approach first
    try:
        # Use pipe operator to create chain: prompt | llm
        reduce_chain = reduce_prompt | llm
        
        # Invoke the chain asynchronously
        result = await reduce_chain.ainvoke({
            "sections": sections_text,
            "company_name": company_name
        })
        
        # Extract the combined text
        if hasattr(result, 'content'):
            return result.content.strip()
        elif isinstance(result, str):
            return result.strip()
        else:
            return str(result).strip()
    
    except Exception as e:
        # Fallback: Use prompt formatting and direct LLM call
        try:
            print(f"⚠️ LCEL approach failed, using fallback: {e}")
            
            # Format messages manually
            formatted_messages = reduce_prompt.format_messages(
                sections=sections_text,
                company_name=company_name
            )
            
            # Build prompt text from messages for Ollama
            prompt_parts = []
            for msg in formatted_messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                    if hasattr(msg, 'type'):
                        if msg.type == 'system':
                            prompt_parts.append(f"System: {content}")
                        elif msg.type == 'human':
                            prompt_parts.append(f"Human: {content}")
                        else:
                            prompt_parts.append(content)
                    else:
                        prompt_parts.append(content)
                else:
                    prompt_parts.append(str(msg))
            
            prompt_text = "\n\n".join(prompt_parts)
            
            # Call LLM - Ollama supports both sync and async
            # Check if LLM has ainvoke method
            if hasattr(llm, 'ainvoke'):
                result = await llm.ainvoke(prompt_text)
            elif hasattr(llm, 'apredict'):
                result = await llm.apredict(prompt_text)
            elif hasattr(llm, '__call__'):
                # Use asyncio.to_thread for sync LLM
                result = await asyncio.to_thread(llm, prompt_text)
            else:
                # Last resort: try invoke directly
                result = await asyncio.to_thread(lambda: llm.invoke(prompt_text))
            
            # Extract text from result
            if hasattr(result, 'content'):
                return result.content.strip()
            elif isinstance(result, dict):
                return result.get('text', result.get('content', str(result))).strip()
            elif isinstance(result, str):
                return result.strip()
            else:
                return str(result).strip()
        
        except Exception as e2:
            print(f"⚠️ Error in reduce phase fallback: {e2}")
            import traceback
            traceback.print_exc()
            # Final fallback: just concatenate sections with transitions
            return "\n\n".join([
                f"## {section.title}\n\n{section.content}"
                for section in sections
            ])


# --------------------------
# Generate narrative article
# --------------------------

async def generate_article(
    company_name: str,
    use_map_reduce: bool = True
) -> ResearchReportResponse:
    """
    Generate a Seeking Alpha-style narrative article for the company.
    
    Args:
        company_name: Company name matching the vectorstore
        use_map_reduce: If True, use map-reduce approach; otherwise use single-pass
    
    Returns:
        ResearchReportResponse object containing the complete report
    """
    try:
        print("📝 Generating narrative article...")

        # Load retriever
        retriever = load_company_retriever(company_name)

        if use_map_reduce:
            # MAP-REDUCE APPROACH
            print("🔄 Using Map-Reduce approach for report generation\n")
            
            # MAP Phase: Generate all sections in parallel
            sections = await generate_all_sections(company_name, retriever, REPORT_SECTIONS)
            
            if not sections:
                raise ValueError("No sections were generated successfully.")
            
            # Collect all unique sources from all sections
            all_sources = []
            for section in sections:
                all_sources.extend(section.sources_used)
            unique_sources = list(set(all_sources))
            
            # REDUCE Phase: Combine sections into final article
            combined_content = await combine_sections(company_name, sections)
            
            # Create a single combined section for the final report
            final_section = ReportSection(
                title="Investment Article",
                content=combined_content,
                sources_used=unique_sources
            )
            
            # Generate the full report markdown
            generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            full_report = f"""# {final_section.title}

**Company:** {company_name}
**Generated:** {generated_at}
**Sources Used:** {len(unique_sources)}
**Generation Method:** Map-Reduce ({len(sections)} sections)

---

{final_section.content}

---

## Sources

"""
            # Add unique sources
            for source in unique_sources:
                full_report += f"- {source}\n"
            
            # Return response with both individual sections and combined result
            return ResearchReportResponse(
                company_name=company_name,
                generated_at=generated_at,
                sections=[final_section],  # Final combined section
                full_report=full_report
            )
        
        else:
            # SINGLE-PASS APPROACH (original implementation)
            print("🔄 Using single-pass approach for report generation\n")
            
            # Create the document chain with the article prompt
            document_chain = create_stuff_documents_chain(llm, article_prompt)
            
            # Create the retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Generate the narrative article
            query = f"Write an investment article for {company_name}"
            
            result = await retrieval_chain.ainvoke({"input": query})

            # Extract the answer and source documents
            result_text = result.get("answer", "")
            retrieved_docs = result.get("context", [])

            # Create the report section
            section = ReportSection(
                title="Investment Article",
                content=result_text,
                sources_used=[doc.metadata.get("source", "unknown") for doc in retrieved_docs]
            )

            # Generate the full report markdown
            generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            full_report = f"""# {section.title}

**Company:** {company_name}
**Generated:** {generated_at}
**Sources Used:** {len(section.sources_used)}
**Generation Method:** Single-Pass

---

{section.content}

---

## Sources

"""
            # Add unique sources
            for source in set(section.sources_used):
                full_report += f"- {source}\n"

            # Create and return the complete response
            return ResearchReportResponse(
                company_name=company_name,
                generated_at=generated_at,
                sections=[section],
                full_report=full_report
            )

    except Exception as e:
        import traceback
        print(f"❌ Cannot generate narrative report: {e}")
        traceback.print_exc()
        
        # Return error response
        error_section = ReportSection(
            title="Investment Article",
            content=f"Error generating narrative article: {str(e)}",
            sources_used=[]
        )
        generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return ResearchReportResponse(
            company_name=company_name,
            generated_at=generated_at,
            sections=[error_section],
            full_report=f"# {error_section.title}\n\n**Company:** {company_name}\n**Generated:** {generated_at}\n\n---\n\n{error_section.content}\n"
        )


# --------------------------
# Save report 
# --------------------------
def save_report(report: ResearchReportResponse, output_dir: str = "output/research_reports") -> str:
    """
    Save the report to a markdown file.
    
    Args:
        report: ResearchReportResponse object containing the complete report
        output_dir: Directory to save the report
    
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    company_slug = report.company_name.replace(' ', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{company_slug}_narrative_report_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Write the full report (already formatted with metadata)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report.full_report)
    
    return filepath

# --------------------------
# CLI entry point
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate narrative investment article from company annual reports",
        epilog="""
Examples:
  # Generate report using map-reduce (default)
  python -m app.services.report Victoria_Secret_2024
  
  # Generate report using single-pass approach
  python -m app.services.report Victoria_Secret_2024 --single-pass
  
  # Generate and print without saving
  python -m app.services.report Victoria_Secret_2024 --no-save --print
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("company_name", nargs="?", default="Victoria_Secret_2024")
    parser.add_argument("--output", "-o", default="output/research_reports")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--print", "-p", action="store_true")
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Use single-pass approach instead of map-reduce (default: map-reduce)"
    )
    args = parser.parse_args()

    try:
        use_map_reduce = not args.single_pass
        report = asyncio.run(generate_article(args.company_name, use_map_reduce=use_map_reduce))

        if not args.no_save:
            filepath = save_report(report, args.output)
            print(f"{'='*60}\n💾 Report saved to: {filepath}\n{'='*60}")

        if args.print or args.no_save:
            print("\n" + "="*60)
            print("FULL REPORT")
            print("="*60 + "\n")
            print(report.full_report)

        # Calculate statistics from all sections
        total_sources = sum(len(section.sources_used) for section in report.sections)
        all_sources = [source for section in report.sections for source in section.sources_used]
        total_content_length = sum(len(section.content) for section in report.sections)
        total_words = sum(len(section.content.split()) for section in report.sections)
        
        print(f"\n📊 Report Statistics:")
        print(f"   • Company: {report.company_name}")
        print(f"   • Generated at: {report.generated_at}")
        print(f"   • Number of sections: {len(report.sections)}")
        print(f"   • Total sources used: {total_sources}")
        print(f"   • Unique sources: {len(set(all_sources))}")
        print(f"   • Report length: {total_content_length:,} characters")
        print(f"   • Word count: {total_words:,} words")

    except KeyboardInterrupt:
        print("\n⚠️ Report generation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
