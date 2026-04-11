# TorchJD Governance

This document defines the governance structure and decision-making process for the TorchJD project.

## Project Ownership

The TorchJD project is the property of SimplexLab. SimplexLab has full authority over the project, including itsdirection, governance structure, and major decisions. Maintainers are typically members of SimplexLab and are responsible for day-to-day operations, code reviews, and technical decisions.

## Maintainers

TorchJD is maintained by:

- **Valérian Rey** ([@ValerianRey](https://github.com/ValerianRey))
- **Pierre Quinton** ([@PierreQuinton](https://github.com/PierreQuinton))

Maintainers are responsible for:
- Reviewing and merging pull requests
- Managing releases
- Setting project direction and priorities
- Ensuring code quality and consistency

## Decision Making

### Technical Decisions

Most technical decisions are made through the pull request process:

1. **Minor changes** (bug fixes, documentation, small improvements): Require approval from at least one maintainer
2. **Significant changes** (new features, API changes, refactoring): Should be discussed in an issue first, then require approval from at least one maintainer
3. **Major changes** (breaking changes, architectural decisions): Should be discussed in an issue or discussion thread and require consensus from all maintainers

### Pull Request Process

1. Contributors submit pull requests following the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md)
2. Maintainers review the code for correctness, style, and alignment with project goals
3. Once approved, any maintainer can merge the pull request
4. All pull requests must pass CI checks before being merged

### Consensus

For major decisions, maintainers aim for consensus. SimplexLab operates as a democratic decision-making body. If consensus among maintainers cannot be reached:
- The decision may be postponed for further discussion
- If a decision must be made, SimplexLab resolves the consensus based on the expertise of all maintainers relevant to the discussion as well as all people involved in the discussion

## Release Process

Releases are managed by maintainers following the process described in [CONTRIBUTING.md](CONTRIBUTING.md):

1. Ensure all tests pass
2. Update the changelog
3. Update the version number
4. Create a release on GitHub
5. Verify deployment to PyPI

## Adding Maintainers

New maintainers may be added when:
- They have made significant, sustained contributions to the project
- They demonstrate understanding of the project's goals and coding standards
- They are committed to the long-term maintenance of the project

New maintainers must be approved by SimplexLab, based on the report and recommendation of all existing maintainers.

## Conflict Resolution

Conflicts are resolved through discussion:
1. Issues should first be discussed in the relevant issue or pull request
2. If unresolved, maintainers discuss privately to reach consensus
3. If maintainers cannot reach consensus, SimplexLab has the final authority to resolve the conflict
4. The goal is always to find the best solution for the project and its users

## Code of Conduct

This project follows the [Linux Foundation Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).

## Changes to Governance

Changes to this governance document can only be made upon request from SimplexLab, which defines when and how such changes are possible.
