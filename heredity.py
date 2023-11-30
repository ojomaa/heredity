import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    person_probability = {}

    # Iterate over each person
    for name in people:

        # Collect data (mother, father, unconditional probability, trait probability)
        gene_number = (2 if name in two_genes else (1 if name in one_gene else 0))
        trait = (True if name in have_trait else False)

        unconditional_prob = PROBS["gene"][gene_number]
        trait_prob = PROBS["trait"][gene_number][trait]
        mother = people[name]['mother']
        father = people[name]['father']

        # If no mother or father, probability is standard
        if mother is None and father is None:
            probability = unconditional_prob * trait_prob
            person_probability[name] = probability

        # If child has mother or father, calculate conditional probability
        else:
            mother_prob = (PROBS["mutation"] if mother not in one_gene and mother not in two_genes else(0.5 if mother in one_gene else 1-PROBS["mutation"]))
            father_prob = (PROBS["mutation"] if father not in one_gene and father not in two_genes else(0.5 if father in one_gene else 1-PROBS["mutation"]))
            not_mother = 1 - mother_prob
            not_father = 1 - father_prob

            if gene_number == 0:
                probability = not_father * not_mother
            elif gene_number == 1:
                probability = (mother_prob * not_father) + (father_prob * not_mother) 
            elif gene_number == 2:
                probability = mother_prob * father_prob
            
            # Calculate child probability 
            probability = probability * PROBS["trait"][gene_number][trait]
            person_probability[name] = probability
                
    
    # Calculate joint probability
    combined_probability = 1
    for value in person_probability.values():
        combined_probability *= value

    # Return Joint Probability
    return combined_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Iterate over each person
    for person in probabilities:

        # Assign gene number and trait
        gene_number = (2 if person in two_genes else (1 if person in one_gene else 0))
        trait = (True if person in have_trait else False)

        # Add joint probability to gene and trait
        probabilities[person]['gene'][gene_number] += p
        probabilities[person]['trait'][trait] += p



def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        sum_gene_values = sum(probabilities[person]['gene'].values())
        sum_trait_values = sum(probabilities[person]['trait'].values())

        normalized_gene_values = {key: value / sum_gene_values for key, value in probabilities[person]['gene'].items()}
        normalized_trait_values = {key: value / sum_trait_values for key, value in probabilities[person]['trait'].items()}

        probabilities[person]['gene'] = normalized_gene_values
        probabilities[person]['trait'] = normalized_trait_values


if __name__ == "__main__":
    main()
